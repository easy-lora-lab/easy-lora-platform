import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanovllm.layers.layernorm import LayerNorm
from nanovllm.layers.linear import MatmulLinear
from nanovllm.models.rwkv7_ops import (
    CMIX_B1T1_NOFC,
    CMIX_NOFC_T512_MIN_ROWS,
    CMIX_ROWS2_NOFC,
    _linear_dispatch,
    _is_contiguous_in_order,
    _wkv7_one_batch_inplace_by_slot_runs,
    _wkv7_one_batch_out_by_slot_runs,
    can_use_lowrank_fused,
    can_use_lowrank_out_fused,
    ensure_rwkv7_fast_loaded,
    linear_orig_layout,
    select_path,
)
from nanovllm.utils.context import get_context


_FAST_TRACE = os.getenv("NANOVLLM_RWKV7_FAST_TRACE", "").lower() not in ("", "0", "false", "off", "no")


class RWKV7Attention(nn.Module):
    """RWKV-7 "Goose" linear attention implementation."""

    def __init__(self, layer_idx: int, hidden_size: int, num_heads: int, head_dim: int, config=None):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.att_tokenshift_cache = None
        self.state_cache = None

        self._fast_trace_shapes = set()
        self.rkv_weight = None
        self.receptance_proj = MatmulLinear(hidden_size, hidden_size, weight_layout="out_in")
        self.key_proj = MatmulLinear(hidden_size, hidden_size, weight_layout="out_in")
        self.value_proj = MatmulLinear(hidden_size, hidden_size, weight_layout="out_in")
        self.output_proj = MatmulLinear(hidden_size, hidden_size, weight_layout="out_in")
        w_rank = int(getattr(config, "decay_low_rank_dim", 128))
        a_rank = int(getattr(config, "a_low_rank_dim", 128))
        v_rank = int(getattr(config, "v_low_rank_dim", 96))
        g_rank = int(getattr(config, "gate_low_rank_dim", 480))
        self.w_rank = w_rank
        self.a_rank = a_rank
        self.v_rank = v_rank
        self.g_rank = g_rank
        self.w1_proj = MatmulLinear(hidden_size, w_rank, weight_layout="out_in")
        self.w2_proj = MatmulLinear(w_rank, hidden_size, weight_layout="out_in")
        self.a1_proj = MatmulLinear(hidden_size, a_rank, weight_layout="out_in")
        self.a2_proj = MatmulLinear(a_rank, hidden_size, weight_layout="out_in")
        self.v1_proj = MatmulLinear(hidden_size, v_rank, weight_layout="out_in")
        self.v2_proj = MatmulLinear(v_rank, hidden_size, weight_layout="out_in")
        self.g1_proj = MatmulLinear(hidden_size, g_rank, weight_layout="out_in")
        self.g2_proj = MatmulLinear(g_rank, hidden_size, weight_layout="out_in")

    @property
    def receptance_weight(self):
        return self.receptance_proj.weight

    @property
    def key_weight(self):
        return self.key_proj.weight

    @property
    def value_weight(self):
        return self.value_proj.weight

    @property
    def output_weight(self):
        return self.output_proj.weight

    @property
    def w1(self):
        return self.w1_proj.weight

    @property
    def w2(self):
        return self.w2_proj.weight

    @property
    def a1(self):
        return self.a1_proj.weight

    @property
    def a2(self):
        return self.a2_proj.weight

    @property
    def v1(self):
        return self.v1_proj.weight

    @property
    def v2(self):
        return self.v2_proj.weight

    @property
    def g1(self):
        return self.g1_proj.weight

    @property
    def g2(self):
        return self.g2_proj.weight

    def _ensure_fast_input(self, x: torch.Tensor) -> None:
        if not x.is_cuda or x.dtype != torch.float16:
            raise RuntimeError("RWKV7 faster3a attention requires CUDA fp16 tensors")
        ensure_rwkv7_fast_loaded(self.head_dim)

    def _contiguous_slot_window(self, slot_mapping_in: torch.Tensor, slot_mapping_out: torch.Tensor):
        if (
            slot_mapping_in is None
            or slot_mapping_out is None
            or slot_mapping_in.numel() == 0
            or slot_mapping_in.numel() != slot_mapping_out.numel()
            or not _is_contiguous_in_order(slot_mapping_in)
            or not _is_contiguous_in_order(slot_mapping_out)
        ):
            return None
        slot_in_start = int(slot_mapping_in[0].item())
        slot_out_start = int(slot_mapping_out[0].item())
        slot_count = int(slot_mapping_in.numel())
        return slot_in_start, slot_out_start, slot_count

    def _fast_linear_orig(self, x: torch.Tensor, proj: MatmulLinear, path, group: str) -> torch.Tensor:
        return linear_orig_layout(x, proj.weight, path, group)

    def _fast_linear_rank_in(self, x: torch.Tensor, proj: MatmulLinear, weight_t: torch.Tensor | None, rows: int) -> torch.Tensor:
        if weight_t is not None and proj.output_size == 128 and rows <= 7:
            return torch.ops.rwkv7_v3a_ops.linear_t_f16(x.contiguous(), weight_t)
        return torch.ops.rwkv7_v3a_ops.linear_f16_orig(x.contiguous(), proj.weight)

    def _fast_linear_rank_out(self, x: torch.Tensor, proj: MatmulLinear, weight_t: torch.Tensor | None, rows: int) -> torch.Tensor:
        if weight_t is not None and proj.input_size == 128 and self.hidden_size >= 1024 and rows <= 4:
            return torch.ops.rwkv7_v3a_ops.linear_t_f16(x.contiguous(), weight_t)
        return torch.ops.rwkv7_v3a_ops.linear_f16_orig(x.contiguous(), proj.weight)

    def _fast_linear_rank_out_act(self, x: torch.Tensor, proj: MatmulLinear, weight_t: torch.Tensor | None, rows: int, act: int) -> torch.Tensor:
        if weight_t is not None and proj.input_size == 128 and self.hidden_size >= 1024 and rows <= 4:
            return torch.ops.rwkv7_v3a_ops.linear_t_act_f16(x.contiguous(), weight_t, act)
        ops = torch.ops.rwkv7_fast_ops_fp16
        x = ops.act_tanh(x.contiguous()) if act == 1 else ops.act_sigmoid(x.contiguous())
        return torch.ops.rwkv7_v3a_ops.linear_f16_orig(x.contiguous(), proj.weight)

    def _forward_prefill_batch_same_length_fast(
        self,
        x: torch.Tensor,
        slot_mapping_in: torch.Tensor,
        slot_mapping_out: torch.Tensor,
        v_first: torch.Tensor | None = None,
    ):
        self._ensure_fast_input(x)
        window = self._contiguous_slot_window(slot_mapping_in, slot_mapping_out)
        if window is None:
            raise RuntimeError("RWKV7 faster3a same-length prefill requires contiguous slot mappings")

        bsz, seqlen, hidden = x.shape
        num_heads, head_dim = self.num_heads, self.head_dim
        path = select_path(bsz, seqlen)
        slot_in_start, slot_out_start, slot_count = window
        if slot_count != bsz:
            raise RuntimeError("RWKV7 faster3a same-length prefill requires one state slot per sequence")
        trace_shape = (bsz, seqlen)
        if _FAST_TRACE and trace_shape not in self._fast_trace_shapes:
            self._fast_trace_shapes.add(trace_shape)
            print(
                "[rwkv7-fast] "
                f"att_same_length layer={self.layer_idx} B={bsz} T={seqlen} rows={path.rows} "
                f"batched_rkv={int(path.use_batched_rkv and self.rkv_weight is not None)} "
                "tmix_mix6=1 wkv_seq=1 output_linear=v3a",
                flush=True,
            )
        if slot_in_start != slot_out_start:
            self.att_tokenshift_cache[slot_out_start:slot_out_start + slot_count].copy_(
                self.att_tokenshift_cache[slot_in_start:slot_in_start + slot_count]
            )
        shift_state = self.att_tokenshift_cache[slot_out_start:slot_out_start + slot_count]
        state = self.state_cache[slot_out_start:slot_out_start + slot_count]
        ops = torch.ops.rwkv7_fast_ops_fp16

        xr, xw, xk, xv, xa, xg = ops.tmix_mix6(
            bsz,
            seqlen,
            hidden,
            x.contiguous(),
            shift_state,
            self.x_r,
            self.x_w,
            self.x_k,
            self.x_v,
            self.x_a,
            self.x_g,
        )
        if path.use_batched_rkv and self.rkv_weight is not None:
            rkv = torch.bmm(
                torch.stack((xr.reshape(-1, hidden), xk.reshape(-1, hidden), xv.reshape(-1, hidden))),
                self.rkv_weight,
            )
            r, k, v = [t.view(bsz, seqlen, hidden) for t in rkv.unbind(0)]
        else:
            r = self._fast_linear_orig(xr, self.receptance_proj, path, "att_c2c")
            k = self._fast_linear_orig(xk, self.key_proj, path, "att_c2c")
            v = self._fast_linear_orig(xv, self.value_proj, path, "att_c2c")

        rows = path.rows
        v1 = None
        can_fuse_wag = self.w_rank == self.a_rank == self.g_rank
        can_fuse_wagv = can_fuse_wag and self.g_rank == self.v_rank
        if can_fuse_wagv and can_use_lowrank_fused(rows, hidden) and can_use_lowrank_out_fused(rows, hidden) and self.layer_idx != 0:
            w1, a1, g1, v1 = torch.ops.rwkv7_v3a_ops.linear_wagv_rank_in_f16(
                xw.contiguous(), xa.contiguous(), xg.contiguous(), xv.contiguous(),
                self.w1_t, self.a1_t, self.g1_t, self.v1_t,
            )
        elif can_fuse_wag and can_use_lowrank_fused(rows, hidden):
            w1, a1, g1 = torch.ops.rwkv7_v3a_ops.linear_wag_rank_in_f16(
                xw.contiguous(), xa.contiguous(), xg.contiguous(), self.w1_t, self.a1_t, self.g1_t,
            )
        else:
            w1 = self._fast_linear_rank_in(xw, self.w1_proj, self.w1_t, rows)
            a1 = self._fast_linear_rank_in(xa, self.a1_proj, self.a1_t, rows)
            g1 = self._fast_linear_rank_in(xg, self.g1_proj, self.g1_t, rows)

        v_done = False
        if can_fuse_wagv and can_use_lowrank_out_fused(rows, hidden) and self.layer_idx != 0 and v1 is not None:
            w, a, g, v = torch.ops.rwkv7_v3a_ops.linear_wagv_rank_out_f16(
                w1.contiguous(), a1.contiguous(), g1.contiguous(), v1.contiguous(),
                self.w2_t, self.a2_t, self.g2_t, self.v2_t,
                v.contiguous(), v_first.contiguous(), self.v0,
            )
            v_done = True
        elif can_fuse_wag and can_use_lowrank_out_fused(rows, hidden):
            w, a, g = torch.ops.rwkv7_v3a_ops.linear_wag_rank_out_f16(
                w1.contiguous(), a1.contiguous(), g1.contiguous(), self.w2_t, self.a2_t, self.g2_t,
            )
        else:
            w = self._fast_linear_rank_out_act(w1, self.w2_proj, self.w2_t, rows, 1)
            a = self._fast_linear_rank_out(a1, self.a2_proj, self.a2_t, rows)
            g = self._fast_linear_rank_out_act(g1, self.g2_proj, self.g2_t, rows, 2)

        k, neg_kk, kka = ops.tmix_kk_a_gate(
            bsz, seqlen, hidden, num_heads, k.contiguous(), self.k_k, self.a0, a.contiguous(), self.k_a
        )
        if self.layer_idx == 0:
            v_first = v
        elif not v_done:
            if can_use_lowrank_out_fused(rows, hidden):
                if v1 is None:
                    v1 = self._fast_linear_rank_in(xv, self.v1_proj, self.v1_t, rows)
                v = torch.ops.rwkv7_v3a_ops.linear_t_vres_f16(
                    v1.contiguous(), self.v2_t, v.contiguous(), v_first.contiguous(), self.v0
                )
            else:
                v12 = self._fast_linear_rank_out(
                    self._fast_linear_rank_in(xv, self.v1_proj, self.v1_t, rows),
                    self.v2_proj,
                    self.v2_t,
                    rows,
                )
                v = ops.tmix_vres_gate(bsz, seqlen, hidden, v.contiguous(), v_first.contiguous(), self.v0, v12.contiguous())

        y = torch.empty_like(r)
        elapsed_t = torch.zeros(bsz, device=x.device, dtype=torch.int32)
        if seqlen <= 16:
            torch.ops.rwkv7_wkv_fp16_v2.wkv_seq_w0(
                bsz, seqlen, hidden, num_heads, state, r.contiguous(), w.contiguous(), self.w0,
                k.contiguous(), v.contiguous(), neg_kk.contiguous(), kka.contiguous(), y, elapsed_t,
            )
        else:
            w_raw = ops.add_vec(hidden, w.contiguous(), self.w0)
            torch.ops.rwkv7_wkv_fp16_v2.wkv_seq(
                bsz, seqlen, hidden, num_heads, state, r.contiguous(), w_raw.contiguous(),
                k.contiguous(), v.contiguous(), neg_kk.contiguous(), kka.contiguous(), y, elapsed_t,
            )
        y = ops.tmix_lnx_rkvres_xg(
            bsz, seqlen, hidden, num_heads, y.contiguous(), r.contiguous(), k.contiguous(), v.contiguous(),
            self.r_k, self.ln_x_weight, self.ln_x_bias, g.contiguous(),
        )
        return self._fast_linear_orig(y, self.output_proj, path, "att_c2c"), v_first

    def forward(
        self,
        positions: torch.Tensor,
        x: torch.Tensor,
        is_prefill: bool,
        v_first: torch.Tensor | None = None,
        att_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context = get_context()
        slot_mapping_in = context.slot_mapping_in
        slot_mapping_out = context.slot_mapping_out

        if is_prefill:
            return self._forward_prefill(x, positions, slot_mapping_in, slot_mapping_out, v_first, att_mask)
        return self._forward_decode(x, positions, slot_mapping_in, slot_mapping_out, v_first)

    def _forward_prefill(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        slot_mapping_in: torch.Tensor,
        slot_mapping_out: torch.Tensor,
        v_first: torch.Tensor | None = None,
        att_mask: torch.Tensor | None = None,
    ):
        if x.dim() != 3:
            raise NotImplementedError("RWKV state-cache path only supports batched prefill.")
        if att_mask is None:
            return self._forward_prefill_batch_same_length(x, slot_mapping_in, slot_mapping_out, v_first)
        raise RuntimeError("RWKV7 faster3a prefill requires model_runner bucketed same-length batches")

    def _forward_prefill_batch_same_length(
        self,
        x: torch.Tensor,
        slot_mapping_in: torch.Tensor,
        slot_mapping_out: torch.Tensor,
        v_first: torch.Tensor | None = None,
    ):
        return self._forward_prefill_batch_same_length_fast(x, slot_mapping_in, slot_mapping_out, v_first)

    def _forward_decode(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        slot_mapping_in: torch.Tensor,
        slot_mapping_out: torch.Tensor,
        v_first: torch.Tensor | None = None,
    ):
        bsz, hidden = x.shape
        num_heads, head_dim = self.num_heads, self.head_dim
        path = select_path(bsz, 1)
        context = get_context()
        assume_equal_slots = getattr(context, "force_regular_decode", False)
        self._ensure_fast_input(x)

        x_prev = self.att_tokenshift_cache[slot_mapping_in].to(x.dtype)
        self.att_tokenshift_cache[slot_mapping_out] = x.to(self.att_tokenshift_cache.dtype)
        xx = x_prev - x
        xr = torch.addcmul(x, xx, self.x_r)
        xw = torch.addcmul(x, xx, self.x_w)
        xk = torch.addcmul(x, xx, self.x_k)
        xv = torch.addcmul(x, xx, self.x_v)
        xa = torch.addcmul(x, xx, self.x_a)
        xg = torch.addcmul(x, xx, self.x_g)

        if path.use_batched_rkv and self.rkv_weight is not None:
            r, k, v = torch.bmm(
                torch.stack((xr, xk, xv)),
                self.rkv_weight,
            ).unbind(0)
        else:
            r = self._fast_linear_orig(xr, self.receptance_proj, path, "att_c2c")
            k = self._fast_linear_orig(xk, self.key_proj, path, "att_c2c")
            v = self._fast_linear_orig(xv, self.value_proj, path, "att_c2c")
        w = _linear_dispatch(torch.tanh(_linear_dispatch(xw, self.w1_proj)), self.w2_proj, bias=self.w0)
        a = torch.sigmoid(_linear_dispatch(_linear_dispatch(xa, self.a1_proj), self.a2_proj, bias=self.a0))
        g = _linear_dispatch(torch.sigmoid(_linear_dispatch(xg, self.g1_proj)), self.g2_proj)
        kk = F.normalize((k * self.k_k).view(bsz, num_heads, head_dim), dim=-1, p=2.0).view_as(k)
        k = k * (1 + (a - 1) * self.k_a)
        kka = kk * a

        if self.layer_idx == 0:
            v_first = v
        else:
            assert v_first is not None
            v = v + (v_first - v) * torch.sigmoid(_linear_dispatch(_linear_dispatch(xv, self.v1_proj), self.v2_proj, bias=self.v0))

        if assume_equal_slots or torch.equal(slot_mapping_in, slot_mapping_out):
            y = _wkv7_one_batch_inplace_by_slot_runs(
                self.state_cache,
                slot_mapping_in,
                r,
                w,
                k,
                v,
                -kk,
                kka,
                positions,
            )
        else:
            y = _wkv7_one_batch_out_by_slot_runs(
                self.state_cache,
                slot_mapping_in,
                slot_mapping_out,
                r,
                w,
                k,
                v,
                -kk,
                kka,
                positions,
            )

        y = F.group_norm(y.view_as(r), num_groups=num_heads, weight=self.ln_x_weight, bias=self.ln_x_bias, eps=64e-5)
        y = y + (
            ((r * k * self.r_k).view(-1, num_heads, head_dim).sum(dim=-1, keepdim=True) * v.view(-1, num_heads, head_dim)).view_as(r)
        )
        return self._fast_linear_orig(y * g, self.output_proj, path, "att_c2c"), v_first


class RWKV7FeedForward(nn.Module):
    """RWKV-7 channel mixing (FFN)."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.ffn_tokenshift_cache = None
        self._fast_trace_shapes = set()
        self.key_proj = MatmulLinear(hidden_size, intermediate_size, weight_layout="out_in")
        self.value_proj = MatmulLinear(intermediate_size, hidden_size)

    @property
    def key_weight(self):
        return self.key_proj.weight

    @property
    def value_weight(self):
        return self.value_proj.weight

    def _ensure_fast_input(self, x: torch.Tensor) -> None:
        if not x.is_cuda or x.dtype != torch.float16:
            raise RuntimeError("RWKV7 faster3a FFN requires CUDA fp16 tensors")
        ensure_rwkv7_fast_loaded(64)

    def forward(self, x: torch.Tensor, is_prefill: bool) -> torch.Tensor:
        context = get_context()
        slot_mapping_in = context.slot_mapping_in
        slot_mapping_out = context.slot_mapping_out

        if is_prefill:
            return self._forward_prefill(x, slot_mapping_in, slot_mapping_out)
        return self._forward_decode(x, slot_mapping_in, slot_mapping_out)

    def _forward_prefill(self, x: torch.Tensor, slot_mapping_in: torch.Tensor, slot_mapping_out: torch.Tensor):
        self._ensure_fast_input(x)
        if x.dim() == 3:
            bsz, seqlen, _ = x.shape
            path = select_path(bsz, seqlen)
            if (
                not _is_contiguous_in_order(slot_mapping_in)
                or not _is_contiguous_in_order(slot_mapping_out)
                or slot_mapping_in.numel() != slot_mapping_out.numel()
            ):
                raise RuntimeError("RWKV7 faster3a FFN prefill requires contiguous slot mappings")
            slot_in_start = int(slot_mapping_in[0].item())
            slot_out_start = int(slot_mapping_out[0].item())
            slot_count = int(slot_mapping_in.numel())
            if slot_count != bsz:
                raise RuntimeError("RWKV7 faster3a FFN prefill requires one state slot per sequence")
            trace_shape = (bsz, seqlen)
            if _FAST_TRACE and trace_shape not in self._fast_trace_shapes:
                self._fast_trace_shapes.add(trace_shape)
                print(
                    "[rwkv7-fast] "
                    f"cmix_same_length B={bsz} T={seqlen} rows={path.rows} mode={path.cmix_mode}",
                    flush=True,
                )
            if slot_in_start != slot_out_start:
                self.ffn_tokenshift_cache[slot_out_start:slot_out_start + slot_count].copy_(
                    self.ffn_tokenshift_cache[slot_in_start:slot_in_start + slot_count]
                )
            ops = torch.ops.rwkv7_fast_ops_fp16
            shift_state = self.ffn_tokenshift_cache[slot_out_start:slot_out_start + slot_count]
            mixed = ops.cmix_mix(bsz, seqlen, x.shape[-1], x.contiguous(), shift_state, self.x_k)
            hid = linear_orig_layout(mixed, self.key_proj.weight, path, "ffn_key")
            if path.cmix_mode == CMIX_B1T1_NOFC:
                return ops.cmix_sparse_down_relu_one(
                    x.shape[-1], self.value_proj.weight.size(0), hid.view(-1).contiguous(), self.value_proj.weight
                )
            if path.cmix_mode == CMIX_ROWS2_NOFC:
                ffn_hidden = self.value_proj.weight.size(0)
                if path.rows >= CMIX_NOFC_T512_MIN_ROWS and x.shape[-1] % 512 == 0 and ffn_hidden % 512 == 0:
                    return ops.cmix_sparse_down_relu_rows_t512(
                        bsz, seqlen, x.shape[-1], ffn_hidden, hid.contiguous(), self.value_proj.weight
                    )
                return ops.cmix_sparse_down_relu_rows(
                    bsz, seqlen, x.shape[-1], ffn_hidden, hid.contiguous(), self.value_proj.weight
                )
            return _linear_dispatch(ops.relu_square(hid.contiguous()), self.value_proj)

        raise RuntimeError("RWKV7 faster3a FFN prefill requires batched 3D tensors")

    def _forward_decode(self, x: torch.Tensor, slot_mapping_in: torch.Tensor, slot_mapping_out: torch.Tensor):
        self._ensure_fast_input(x)
        if (
            isinstance(self.key_proj, MatmulLinear)
            and isinstance(self.value_proj, MatmulLinear)
            and self.key_proj.weight_layout == "out_in"
            and self.value_proj.weight_layout == "in_out"
            and _is_contiguous_in_order(slot_mapping_in)
            and _is_contiguous_in_order(slot_mapping_out)
            and slot_mapping_in.numel() == slot_mapping_out.numel()
        ):
            bsz, hidden = x.shape
            slot_in_start = int(slot_mapping_in[0].item())
            slot_out_start = int(slot_mapping_out[0].item())
            slot_count = slot_mapping_in.numel()
            if slot_count == bsz:
                if slot_in_start != slot_out_start:
                    self.ffn_tokenshift_cache[slot_out_start:slot_out_start + slot_count].copy_(
                        self.ffn_tokenshift_cache[slot_in_start:slot_in_start + slot_count]
                    )
                path = select_path(bsz, 1)
                ops = torch.ops.rwkv7_fast_ops_fp16
                shift_state = self.ffn_tokenshift_cache[slot_out_start:slot_out_start + slot_count]
                mixed = ops.cmix_mix(bsz, 1, hidden, x.view(bsz, 1, hidden).contiguous(), shift_state, self.x_k)
                hid = linear_orig_layout(mixed, self.key_proj.weight, path, "ffn_key")
                if path.cmix_mode == CMIX_B1T1_NOFC:
                    return ops.cmix_sparse_down_relu_one(
                        hidden,
                        self.value_proj.weight.size(0),
                        hid.view(-1).contiguous(),
                        self.value_proj.weight,
                    ).view_as(x)
                if path.cmix_mode == CMIX_ROWS2_NOFC:
                    ffn_hidden = self.value_proj.weight.size(0)
                    if path.rows >= CMIX_NOFC_T512_MIN_ROWS and hidden % 512 == 0 and ffn_hidden % 512 == 0:
                        return ops.cmix_sparse_down_relu_rows_t512(
                            bsz, 1, hidden, ffn_hidden, hid.contiguous(), self.value_proj.weight
                        ).view_as(x)
                    return ops.cmix_sparse_down_relu_rows(
                        bsz, 1, hidden, ffn_hidden, hid.contiguous(), self.value_proj.weight
                    ).view_as(x)
                return _linear_dispatch(ops.relu_square(hid.contiguous()), self.value_proj).view_as(x)

        raise RuntimeError("RWKV7 faster3a FFN decode requires contiguous slots and MatmulLinear weights")


class RWKV7Block(nn.Module):
    """RWKV-7 transformer block."""

    def __init__(self, layer_idx: int, hidden_size: int, num_heads: int, head_dim: int, intermediate_size: int, config=None):
        super().__init__()
        self.layer_idx = layer_idx

        self.ln1 = LayerNorm(hidden_size)
        self.att = RWKV7Attention(layer_idx, hidden_size, num_heads, head_dim, config)
        self.ln2 = LayerNorm(hidden_size)
        self.ffn = RWKV7FeedForward(hidden_size, intermediate_size)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        is_prefill: bool,
        v_first: torch.Tensor | None = None,
        att_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not is_prefill:
            context = get_context()
            slot_mapping_in = context.slot_mapping_in
            slot_mapping_out = context.slot_mapping_out

            h = F.layer_norm(x, (self.ln1.hidden_size,), self.ln1.gamma, self.ln1.beta, self.ln1.eps)
            h, v_first = self.att._forward_decode(h, positions, slot_mapping_in, slot_mapping_out, v_first)
            x.add_(h)

            h = F.layer_norm(x, (self.ln2.hidden_size,), self.ln2.gamma, self.ln2.beta, self.ln2.eps)
            h = self.ffn._forward_decode(h, slot_mapping_in, slot_mapping_out)
            x.add_(h)
            return x, v_first

        xx = self.ln1(x)
        if att_mask is not None:
            xx.masked_fill_(att_mask, 0)
        h, v_first = self.att(positions, xx, is_prefill, v_first, att_mask)
        x.add_(h)
        if att_mask is not None:
            x.masked_fill_(att_mask, 0)

        h = self.ln2(x)
        if att_mask is not None:
            h.masked_fill_(att_mask, 0)
        h = self.ffn(h, is_prefill)
        x.add_(h)
        if att_mask is not None:
            x.masked_fill_(att_mask, 0)

        return x, v_first
