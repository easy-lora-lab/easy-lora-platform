import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from nanovllm.layers.linear import MatmulLinear
from nanovllm.ops.rwkv7_cuda import ensure_faster3a_loaded
from nanovllm.utils.context import get_context


RKV_MODE = os.getenv("NANOVLLM_RWKV7_RKV_MODE", "auto")
CMIX_SPARSE = os.getenv("NANOVLLM_RWKV7_CMIX_SPARSE", "no-fc")
LOWRANK_WEIGHT = os.getenv("NANOVLLM_RWKV7_LOWRANK_WEIGHT", "both")
LOWRANK_SUFFIXES = ("att.w1", "att.w2", "att.a1", "att.a2", "att.g1", "att.g2", "att.v1", "att.v2")
LOWRANK_IN_ROWS_T = 7
LOWRANK_OUT_ROWS_T = 4
LOWRANK_FUSED_MIN_C = int(os.getenv("NANOVLLM_RWKV7_LOWRANK_FUSED_MIN_C", "1024"))
CMIX_NOFC_ROW20_MAX_T = int(os.getenv("NANOVLLM_RWKV7_CMIX_NOFC_ROW20_MAX_T", "5"))
CMIX_B1T1_SPARSE = "b1t1_sparse"
CMIX_ROWS2_SPARSE = "rows2_sparse"
CMIX_B1T1_NOFC = "b1t1_nofc"
CMIX_ROWS2_NOFC = "rows2_nofc"
CMIX_DENSE = "dense"
CMIX_NOFC_T512_MIN_ROWS = 8


def parse_orig_linear_groups(text: str) -> set[str]:
    groups = {item.strip() for item in text.replace(",", " ").split() if item.strip()}
    if not groups or groups == {"none"}:
        return set()
    allowed = {"att_c2c", "ffn_key", "head"}
    unknown = groups - allowed
    if unknown:
        raise ValueError(f"unknown orig linear groups: {sorted(unknown)}")
    return groups


ORIG_LINEAR_GROUPS = parse_orig_linear_groups(os.getenv("NANOVLLM_RWKV7_ORIG_LINEAR_GROUPS", "att_c2c,ffn_key,head"))


@dataclass(frozen=True)
class PathConfig:
    rows: int
    use_batched_rkv: bool
    cmix_mode: str


def cmix_nofc_max_rows() -> int:
    return 19


def cmix_nofc_row20_max_t() -> int:
    return CMIX_NOFC_ROW20_MAX_T


def use_orig_linear(group: str) -> bool:
    return group in ORIG_LINEAR_GROUPS


def is_lowrank_weight(key: str) -> bool:
    return key.endswith(LOWRANK_SUFFIXES)


def can_use_lowrank_fused(rows: int, hidden_size: int | None = None) -> bool:
    if hidden_size is not None and hidden_size < LOWRANK_FUSED_MIN_C:
        return False
    return rows <= LOWRANK_IN_ROWS_T


def can_use_lowrank_out_fused(rows: int, hidden_size: int | None = None) -> bool:
    if hidden_size is not None and hidden_size < LOWRANK_FUSED_MIN_C:
        return False
    return rows <= LOWRANK_OUT_ROWS_T


def is_att_c2c_weight(key: str) -> bool:
    return ".att." in key and key.endswith(("receptance.weight", "key.weight", "value.weight", "output.weight"))


def is_orig_linear_weight(key: str) -> bool:
    return (
        (use_orig_linear("att_c2c") and is_att_c2c_weight(key))
        or (use_orig_linear("ffn_key") and ".ffn.key.weight" in key)
        or (use_orig_linear("head") and key == "head.weight")
    )


def select_path(B: int, T: int) -> PathConfig:
    rows = B * T
    if CMIX_SPARSE == "off":
        cmix_mode = CMIX_DENSE
    elif CMIX_SPARSE == "no-fc":
        use_nofc = rows <= cmix_nofc_max_rows() or (rows == 20 and T <= cmix_nofc_row20_max_t())
        cmix_mode = CMIX_B1T1_NOFC if rows == 1 else (CMIX_ROWS2_NOFC if use_nofc else CMIX_DENSE)
    elif rows == 1:
        cmix_mode = CMIX_B1T1_SPARSE
    elif rows == 2:
        cmix_mode = CMIX_ROWS2_NOFC
    else:
        cmix_mode = CMIX_DENSE

    if RKV_MODE == "auto":
        use_batched_rkv = rows == 1 or 4 <= rows <= 64
    elif RKV_MODE == "on":
        use_batched_rkv = True
    else:
        use_batched_rkv = False
    if use_orig_linear("att_c2c"):
        use_batched_rkv = False
    return PathConfig(rows=rows, use_batched_rkv=use_batched_rkv, cmix_mode=cmix_mode)


def ensure_rwkv7_fast_loaded(head_dim: int) -> None:
    if head_dim != 64:
        raise RuntimeError(f"faster3a RWKV7 WKV kernels require head_dim=64, got {head_dim}")
    ensure_faster3a_loaded()


def _wkv7_fast_one_batch(
    state_in: torch.Tensor,
    state_out: torch.Tensor,
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kk: torch.Tensor,
    kka: torch.Tensor,
    elapsed_t: torch.Tensor,
) -> torch.Tensor:
    ensure_faster3a_loaded("fp16")
    if state_out.data_ptr() != state_in.data_ptr():
        state_out.copy_(state_in)
    bsz, hidden = r.shape
    num_heads = state_out.size(1)
    y = torch.empty_like(r, memory_format=torch.contiguous_format)
    torch.ops.rwkv7_wkv_fp16_v2.wkv_one(
        bsz,
        hidden,
        num_heads,
        state_out,
        r.contiguous(),
        w.contiguous(),
        k.contiguous(),
        v.contiguous(),
        kk.contiguous(),
        kka.contiguous(),
        y,
        elapsed_t.to(device=r.device, dtype=torch.int32).contiguous(),
    )
    return y


def _wkv7_fast_seq_batch(
    state_in: torch.Tensor,
    state_out: torch.Tensor,
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kk: torch.Tensor,
    kka: torch.Tensor,
    elapsed_t: torch.Tensor,
) -> torch.Tensor:
    ensure_faster3a_loaded("fp16")
    if state_out.data_ptr() != state_in.data_ptr():
        state_out.copy_(state_in)
    bsz, seqlen, hidden = r.shape
    num_heads = state_out.size(1)
    y = torch.empty_like(r, memory_format=torch.contiguous_format)
    torch.ops.rwkv7_wkv_fp16_v2.wkv_seq(
        bsz,
        seqlen,
        hidden,
        num_heads,
        state_out,
        r.contiguous(),
        w.contiguous(),
        k.contiguous(),
        v.contiguous(),
        kk.contiguous(),
        kka.contiguous(),
        y,
        elapsed_t.to(device=r.device, dtype=torch.int32).contiguous(),
    )
    return y


def _linear_f16(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.ops.rwkv7_v3a_ops.linear_f16(x.contiguous(), weight)


def _linear_f16_orig(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.ops.rwkv7_v3a_ops.linear_f16_orig(x.contiguous(), weight)


def linear_orig_layout(x: torch.Tensor, weight: torch.Tensor, path: PathConfig, group: str) -> torch.Tensor:
    if not use_orig_linear(group):
        return torch.ops.rwkv7_v3a_ops.linear_f16(x.contiguous(), weight)
    c = x.shape[-1]
    rows = path.rows
    if rows == 1:
        if group == "ffn_key":
            if c == 2560:
                return torch.ops.rwkv7_v3a_ops.linear_orig_rows_exact_f16(x.contiguous(), weight, 128, 2, True)
            return torch.ops.rwkv7_v3a_ops.linear_orig_rows_exact_f16(x.contiguous(), weight, 128, 2, c <= 1024)
        return torch.ops.rwkv7_v3a_ops.linear_orig_rows_exact_f16(x.contiguous(), weight, 128, 2, group != "att_c2c" or c < 2048)
    if rows == 2:
        if group == "att_c2c":
            return torch.ops.rwkv7_v3a_ops.linear_orig_rows_exact_f16(x.contiguous(), weight, 64, 2, True)
        if group == "ffn_key":
            if c == 2560:
                return torch.ops.rwkv7_v3a_ops.linear_orig_rows_exact_f16(x.contiguous(), weight, 128, 2, False)
            if c < 4096:
                return torch.ops.rwkv7_v3a_ops.linear_orig_rows_exact_f16(x.contiguous(), weight, 64, 2, True)
            return torch.ops.rwkv7_v3a_ops.linear_orig_rows_exact_f16(x.contiguous(), weight, 128, 2, False)
        if group == "head" and c == 2560:
            return torch.ops.rwkv7_v3a_ops.linear_orig_rows_exact_f16(x.contiguous(), weight, 128, 2, False)
        return torch.ops.rwkv7_v3a_ops.linear_orig_rows_exact_f16(x.contiguous(), weight, 64, 2, True)
    if group == "head":
        if rows >= 1024:
            return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 128, 0)
        if rows >= 512:
            return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 2)
        if rows >= 384:
            return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 128, 2)
        if rows >= 256:
            return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 1)
        if rows >= 192:
            return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 128, 0)
        if rows >= 160:
            return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 0)
        if rows >= 128:
            return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 128, 0)
        if rows >= 96:
            return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 1)
    if group == "att_c2c":
        if rows >= 1024:
            return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 4)
        if rows >= 768:
            return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 0)
        if rows >= 512:
            return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 1)
        if rows >= 384:
            return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 128, 2)
        if rows >= 256:
            return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 4)
        if rows >= 192:
            return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 0)
        if rows >= 160:
            return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 128, 1)
        if rows >= 112:
            return torch.ops.rwkv7_v3a_ops.linear_f16_orig(x.contiguous(), weight)
        if rows >= 96:
            return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 5)
        if rows >= 72:
            return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 0)
        if rows >= 48:
            return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 6)
        if rows >= 32:
            return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 0)
        if rows >= 24:
            return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 6)
        if rows >= 12:
            return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 0)
        if rows >= 5:
            return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 2)
    if group == "ffn_key":
        if rows >= 1024:
            return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 0)
        if rows >= 768:
            return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 1)
        if rows >= 512:
            return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 128, 3)
        if rows >= 384:
            return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 0)
        if rows >= 256:
            return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 128, 4)
        if rows >= 192:
            return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 1)
        if rows >= 160:
            return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 2)
        if rows >= 128:
            return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 0)
        if rows >= 112:
            return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 3)
        if rows >= 96:
            return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 1)
        if rows >= 72:
            return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 128, 1)
        if rows >= 48:
            return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 1)
        if rows >= 12:
            return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 0)
        if rows in (5, 6):
            return torch.ops.rwkv7_v3a_ops.linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 1)
    return torch.ops.rwkv7_v3a_ops.linear_f16_orig(x.contiguous(), weight)


def _linear_batched_rkv(
    xr: torch.Tensor,
    xk: torch.Tensor,
    xv: torch.Tensor,
    rkv_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    c = xr.shape[-1]
    flat = torch.stack((xr.reshape(-1, c), xk.reshape(-1, c), xv.reshape(-1, c)))
    rkv = torch.bmm(flat, rkv_weight)
    return tuple(t.reshape_as(xr) for t in rkv.unbind(0))


_MAX_NONCONTIGUOUS_STATE_GATHER_ROWS = max(
    1,
    int(os.getenv("NANOVLLM_MAX_NONCONTIGUOUS_STATE_GATHER_ROWS", "4")),
)


def _linear_dispatch(
    x: torch.Tensor,
    proj_or_weight,
    bias: torch.Tensor | None = None,
    path: PathConfig | None = None,
    group: str | None = None,
):
    if isinstance(proj_or_weight, torch.Tensor):
        y = _linear_f16(x, proj_or_weight)
        return y if bias is None else y + bias
    if isinstance(proj_or_weight, MatmulLinear):
        proj_bias = bias if bias is not None else proj_or_weight.bias
        if proj_or_weight.weight_layout == "out_in":
            y = (
                linear_orig_layout(x, proj_or_weight.weight, path, group)
                if path is not None and group is not None
                else _linear_f16_orig(x, proj_or_weight.weight)
            )
        else:
            y = _linear_f16(x, proj_or_weight.weight)
        return y if proj_bias is None else y + proj_bias
    proj_bias = bias if bias is not None else proj_or_weight.bias
    y = _linear_f16(x, proj_or_weight.weight)
    return y if proj_bias is None else y + proj_bias


def _rwkv7_decode_block_batch_contiguous(
    x: torch.Tensor,
    att_tokenshift_cache_in: torch.Tensor,
    att_tokenshift_cache_out: torch.Tensor,
    state_cache_in: torch.Tensor,
    state_cache_out: torch.Tensor,
    ffn_tokenshift_cache_in: torch.Tensor,
    ffn_tokenshift_cache_out: torch.Tensor,
    positions: torch.Tensor,
    v_first: torch.Tensor | None,
    layer_idx: int,
    num_heads: int,
    head_dim: int,
    x_r: torch.Tensor,
    x_w: torch.Tensor,
    x_k: torch.Tensor,
    x_v: torch.Tensor,
    x_a: torch.Tensor,
    x_g: torch.Tensor,
    w0: torch.Tensor,
    w1_proj,
    w2_proj,
    a0: torch.Tensor,
    a1_proj,
    a2_proj,
    v0: torch.Tensor,
    v1_proj,
    v2_proj,
    g1_proj,
    g2_proj,
    k_k: torch.Tensor,
    k_a: torch.Tensor,
    r_k: torch.Tensor,
    receptance_proj,
    key_proj,
    value_proj,
    output_proj,
    ln_x_weight: torch.Tensor,
    ln_x_bias: torch.Tensor,
    ln1_gamma: torch.Tensor,
    ln1_beta: torch.Tensor,
    ln1_eps: float,
    ln2_gamma: torch.Tensor,
    ln2_beta: torch.Tensor,
    ln2_eps: float,
    ffn_x_k: torch.Tensor,
    ffn_key_proj,
    ffn_value_proj,
    decode_tokenshift_scratch: torch.Tensor,
):
    bsz, c = x.shape
    if not x.is_cuda or x.dtype != torch.float16:
        raise RuntimeError("RWKV7 faster3a decode requires CUDA fp16 tensors")
    ensure_rwkv7_fast_loaded(head_dim)
    att_inplace = att_tokenshift_cache_in.data_ptr() == att_tokenshift_cache_out.data_ptr()
    ffn_inplace = ffn_tokenshift_cache_in.data_ptr() == ffn_tokenshift_cache_out.data_ptr()
    h = F.layer_norm(x, (c,), ln1_gamma, ln1_beta, ln1_eps)
    if att_inplace:
        xx = att_tokenshift_cache_in.to(dtype=h.dtype) - h
        att_tokenshift_cache_out.copy_(h)
    else:
        h_cache = h.to(att_tokenshift_cache_out.dtype)
        x_prev = decode_tokenshift_scratch[:bsz]
        x_prev.copy_(att_tokenshift_cache_in)
        att_tokenshift_cache_out.copy_(h_cache)
        xx = x_prev - h
    xr = torch.addcmul(h, xx, x_r)
    xw = torch.addcmul(h, xx, x_w)
    xk = torch.addcmul(h, xx, x_k)
    xv = torch.addcmul(h, xx, x_v)
    xa = torch.addcmul(h, xx, x_a)
    xg = torch.addcmul(h, xx, x_g)

    r = _linear_dispatch(xr, receptance_proj)
    w = _linear_dispatch(torch.tanh(_linear_dispatch(xw, w1_proj)), w2_proj, bias=w0)
    k = _linear_dispatch(xk, key_proj)
    v = _linear_dispatch(xv, value_proj)
    a = torch.sigmoid(_linear_dispatch(_linear_dispatch(xa, a1_proj), a2_proj, bias=a0))
    g = _linear_dispatch(torch.sigmoid(_linear_dispatch(xg, g1_proj)), g2_proj)

    kk = F.normalize((k * k_k.view(1, 1, c)).view(bsz, num_heads, head_dim), dim=-1, p=2.0).view(bsz, c)
    k = k * (1 + (a - 1) * k_a.view(1, c))
    kka = kk * a

    if layer_idx == 0:
        v_first = v
    else:
        assert v_first is not None
        v = v + (v_first - v) * torch.sigmoid(_linear_dispatch(_linear_dispatch(xv, v1_proj), v2_proj, bias=v0))

    y = _wkv7_fast_one_batch(
        state_cache_in,
        state_cache_out,
        r,
        w,
        k,
        v,
        -kk,
        kka,
        positions,
    )
    y = F.group_norm(y.view(bsz, c), num_groups=num_heads, weight=ln_x_weight, bias=ln_x_bias, eps=64e-5)
    y = y + (
        ((r * k * r_k.view(1, c)).view(bsz, num_heads, head_dim).sum(dim=-1, keepdim=True) * v.view(bsz, num_heads, head_dim)).view(bsz, c)
    )
    y = _linear_dispatch(y * g, output_proj)
    x.add_(y)

    h2 = F.layer_norm(x, (c,), ln2_gamma, ln2_beta, ln2_eps)
    if (
        isinstance(ffn_key_proj, MatmulLinear)
        and isinstance(ffn_value_proj, MatmulLinear)
        and ffn_key_proj.weight_layout == "out_in"
        and ffn_value_proj.weight_layout == "in_out"
    ):
        if not ffn_inplace:
            ffn_tokenshift_cache_out.copy_(ffn_tokenshift_cache_in)
        path = select_path(bsz, 1)
        ops = torch.ops.rwkv7_fast_ops_fp16
        mixed = ops.cmix_mix(bsz, 1, c, h2.view(bsz, 1, c).contiguous(), ffn_tokenshift_cache_out, ffn_x_k)
        hid = linear_orig_layout(mixed, ffn_key_proj.weight, path, "ffn_key")
        if path.cmix_mode == CMIX_B1T1_NOFC:
            out = ops.cmix_sparse_down_relu_one(
                c,
                ffn_value_proj.weight.size(0),
                hid.view(-1).contiguous(),
                ffn_value_proj.weight,
            )
        elif path.cmix_mode == CMIX_ROWS2_NOFC:
            ffn_hidden = ffn_value_proj.weight.size(0)
            if path.rows >= CMIX_NOFC_T512_MIN_ROWS and c % 512 == 0 and ffn_hidden % 512 == 0:
                out = ops.cmix_sparse_down_relu_rows_t512(bsz, 1, c, ffn_hidden, hid.contiguous(), ffn_value_proj.weight)
            else:
                out = ops.cmix_sparse_down_relu_rows(bsz, 1, c, ffn_hidden, hid.contiguous(), ffn_value_proj.weight)
        else:
            out = _linear_dispatch(ops.relu_square(hid.contiguous()), ffn_value_proj)
        x.add_(out.view(bsz, c))
    else:
        raise RuntimeError("RWKV7 faster3a FFN decode requires MatmulLinear fp16 weights")
    return x, v_first


def _rwkv7_decode_block_one_contiguous(
    x: torch.Tensor,
    att_tokenshift_cache_in: torch.Tensor,
    att_tokenshift_cache_out: torch.Tensor,
    state_cache_in: torch.Tensor,
    state_cache_out: torch.Tensor,
    ffn_tokenshift_cache_in: torch.Tensor,
    ffn_tokenshift_cache_out: torch.Tensor,
    positions: torch.Tensor,
    v_first: torch.Tensor | None,
    layer_idx: int,
    num_heads: int,
    head_dim: int,
    x_r: torch.Tensor,
    x_w: torch.Tensor,
    x_k: torch.Tensor,
    x_v: torch.Tensor,
    x_a: torch.Tensor,
    x_g: torch.Tensor,
    w0: torch.Tensor,
    w1_proj,
    w2_proj,
    a0: torch.Tensor,
    a1_proj,
    a2_proj,
    v0: torch.Tensor,
    v1_proj,
    v2_proj,
    g1_proj,
    g2_proj,
    k_k: torch.Tensor,
    k_a: torch.Tensor,
    r_k: torch.Tensor,
    receptance_proj,
    key_proj,
    value_proj,
    output_proj,
    ln_x_weight: torch.Tensor,
    ln_x_bias: torch.Tensor,
    ln1_gamma: torch.Tensor,
    ln1_beta: torch.Tensor,
    ln1_eps: float,
    ln2_gamma: torch.Tensor,
    ln2_beta: torch.Tensor,
    ln2_eps: float,
    ffn_x_k: torch.Tensor,
    ffn_key_proj,
    ffn_value_proj,
    decode_tokenshift_scratch: torch.Tensor,
):
    if not x.is_cuda or x.dtype != torch.float16:
        raise RuntimeError("RWKV7 faster3a decode requires CUDA fp16 tensors")
    ensure_rwkv7_fast_loaded(head_dim)

    x0 = x[0]
    h = F.layer_norm(x0, (x0.shape[-1],), ln1_gamma, ln1_beta, ln1_eps)
    if att_tokenshift_cache_out.data_ptr() != att_tokenshift_cache_in.data_ptr():
        att_tokenshift_cache_out[0].copy_(att_tokenshift_cache_in[0])
    x_prev = att_tokenshift_cache_out[0]
    xx = x_prev - h
    x_prev.copy_(h)
    xr = torch.addcmul(h, xx, x_r)
    xw = torch.addcmul(h, xx, x_w)
    xk = torch.addcmul(h, xx, x_k)
    xv = torch.addcmul(h, xx, x_v)
    xa = torch.addcmul(h, xx, x_a)
    xg = torch.addcmul(h, xx, x_g)

    r = _linear_dispatch(xr, receptance_proj)
    w = _linear_dispatch(torch.tanh(_linear_dispatch(xw, w1_proj)), w2_proj, bias=w0)
    k = _linear_dispatch(xk, key_proj)
    v = _linear_dispatch(xv, value_proj)
    a = torch.sigmoid(_linear_dispatch(_linear_dispatch(xa, a1_proj), a2_proj, bias=a0))
    g = _linear_dispatch(torch.sigmoid(_linear_dispatch(xg, g1_proj)), g2_proj)
    kk = F.normalize((k * k_k).view(num_heads, head_dim), dim=-1, p=2.0).view_as(k)
    k = k * (1 + (a - 1) * k_a)
    kka = kk * a
    if layer_idx == 0:
        v_first = v
    else:
        assert v_first is not None
        v = v + (v_first - v) * torch.sigmoid(_linear_dispatch(_linear_dispatch(xv, v1_proj), v2_proj, bias=v0))
    y = _wkv7_fast_one_batch(
        state_cache_in,
        state_cache_out,
        r.view(1, -1),
        w.view(1, -1),
        k.view(1, -1),
        v.view(1, -1),
        -kk.view(1, -1),
        kka.view(1, -1),
        positions[0:1],
    ).view(-1)
    y = F.group_norm(y.view(1, -1), num_groups=num_heads, weight=ln_x_weight, bias=ln_x_bias, eps=64e-5).view(-1)
    y = y + ((r * k * r_k).view(1, num_heads, head_dim).sum(dim=-1, keepdim=True) * v.view(1, num_heads, head_dim)).view_as(r)
    y = _linear_dispatch(y * g, output_proj)
    x0.add_(y)

    h2 = F.layer_norm(x0, (x0.shape[-1],), ln2_gamma, ln2_beta, ln2_eps)
    if (
        isinstance(ffn_key_proj, MatmulLinear)
        and isinstance(ffn_value_proj, MatmulLinear)
        and ffn_key_proj.weight_layout == "out_in"
        and ffn_value_proj.weight_layout == "in_out"
    ):
        if ffn_tokenshift_cache_out.data_ptr() != ffn_tokenshift_cache_in.data_ptr():
            ffn_tokenshift_cache_out[0].copy_(ffn_tokenshift_cache_in[0])
        path = select_path(1, 1)
        ops = torch.ops.rwkv7_fast_ops_fp16
        mixed = ops.cmix_mix(1, 1, x0.shape[-1], h2.view(1, 1, -1).contiguous(), ffn_tokenshift_cache_out, ffn_x_k)
        hid = linear_orig_layout(mixed, ffn_key_proj.weight, path, "ffn_key")
        if path.cmix_mode == CMIX_B1T1_NOFC:
            out = ops.cmix_sparse_down_relu_one(
                x0.shape[-1],
                ffn_value_proj.weight.size(0),
                hid.view(-1).contiguous(),
                ffn_value_proj.weight,
            )
        else:
            out = _linear_dispatch(ops.relu_square(hid.contiguous()), ffn_value_proj).view(-1)
        x0.add_(out.view_as(x0))
    else:
        raise RuntimeError("RWKV7 faster3a FFN decode requires MatmulLinear fp16 weights")
    return x, v_first


def _build_slot_runs(slots: torch.Tensor):
    order = torch.argsort(slots)
    sorted_slots = slots[order]
    sorted_slots_list = sorted_slots.tolist()
    runs = []
    start = 0
    n = len(sorted_slots_list)
    while start < n:
        end = start + 1
        while end < n and sorted_slots_list[end] == sorted_slots_list[end - 1] + 1:
            end += 1
        runs.append((start, end, sorted_slots_list[start], sorted_slots_list[end - 1] + 1))
        start = end
    has_duplicates = any(
        sorted_slots_list[i] == sorted_slots_list[i - 1]
        for i in range(1, n)
    )
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(order.numel(), device=order.device, dtype=order.dtype)
    return order, inverse, runs, has_duplicates


def _is_contiguous_in_order(slots: torch.Tensor) -> bool:
    if slots.numel() <= 1:
        return True
    expected = torch.arange(
        int(slots[0].item()),
        int(slots[0].item()) + slots.numel(),
        device=slots.device,
        dtype=slots.dtype,
    )
    return torch.equal(slots, expected)


def _state_rows_for_sorted_run(state_cache: torch.Tensor, sorted_slots: torch.Tensor, start: int, end: int) -> torch.Tensor:
    run_slots = sorted_slots[start:end]
    if _is_contiguous_in_order(run_slots):
        slot_start = int(run_slots[0].item())
        return state_cache[slot_start:slot_start + (end - start)]
    return state_cache[run_slots].contiguous()


def _wkv7_one_batch_inplace_by_slot_runs(
    state_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kk: torch.Tensor,
    kka: torch.Tensor,
    positions: torch.Tensor,
):
    context = get_context()
    if getattr(context, "force_regular_decode", False):
        return _wkv7_fast_one_batch(
            state_cache[slot_mapping],
            state_cache[slot_mapping],
            r,
            w,
            k,
            v,
            kk,
            kka,
            positions,
        )
    if _is_contiguous_in_order(slot_mapping):
        slot_start = int(slot_mapping[0].item())
        slot_end = slot_start + slot_mapping.numel()
        return _wkv7_fast_one_batch(
            state_cache[slot_start:slot_end],
            state_cache[slot_start:slot_end],
            r,
            w,
            k,
            v,
            kk,
            kka,
            positions,
        )

    order, inverse, runs, has_duplicates = _build_slot_runs(slot_mapping)
    if has_duplicates:
        state = state_cache[slot_mapping].contiguous()
        y = _wkv7_fast_one_batch(state, state, r, w, k, v, kk, kka, positions)
        state_cache[slot_mapping] = state
        return y

    r_sorted = r[order]
    w_sorted = w[order]
    k_sorted = k[order]
    v_sorted = v[order]
    kk_sorted = kk[order]
    kka_sorted = kka[order]
    positions_sorted = positions[order]
    y_sorted = torch.empty_like(r_sorted)

    for start, end, slot_start, slot_end in runs:
        y_sorted[start:end] = _wkv7_fast_one_batch(
            state_cache[slot_start:slot_end],
            state_cache[slot_start:slot_end],
            r_sorted[start:end],
            w_sorted[start:end],
            k_sorted[start:end],
            v_sorted[start:end],
            kk_sorted[start:end],
            kka_sorted[start:end],
            positions_sorted[start:end],
        )

    return y_sorted[inverse]


def _wkv7_one_batch_out_by_slot_runs(
    state_cache: torch.Tensor,
    slot_mapping_in: torch.Tensor,
    slot_mapping_out: torch.Tensor,
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kk: torch.Tensor,
    kka: torch.Tensor,
    positions: torch.Tensor,
):
    context = get_context()
    if getattr(context, "force_regular_decode", False):
        return _wkv7_fast_one_batch(
            state_cache[slot_mapping_in],
            state_cache[slot_mapping_out],
            r,
            w,
            k,
            v,
            kk,
            kka,
            positions,
        )
    if _is_contiguous_in_order(slot_mapping_in) and _is_contiguous_in_order(slot_mapping_out):
        slot_in_start = int(slot_mapping_in[0].item())
        slot_out_start = int(slot_mapping_out[0].item())
        slot_count = slot_mapping_in.numel()
        return _wkv7_fast_one_batch(
            state_cache[slot_in_start:slot_in_start + slot_count],
            state_cache[slot_out_start:slot_out_start + slot_count],
            r,
            w,
            k,
            v,
            kk,
            kka,
            positions,
        )

    order, inverse, runs, has_duplicates = _build_slot_runs(slot_mapping_out)
    if has_duplicates:
        state_in = state_cache[slot_mapping_in].contiguous()
        state_out = state_cache[slot_mapping_out].contiguous()
        y = _wkv7_fast_one_batch(state_in, state_out, r, w, k, v, kk, kka, positions)
        state_cache[slot_mapping_out] = state_out
        return y

    slot_in_sorted = slot_mapping_in[order]
    r_sorted = r[order]
    w_sorted = w[order]
    k_sorted = k[order]
    v_sorted = v[order]
    kk_sorted = kk[order]
    kka_sorted = kka[order]
    positions_sorted = positions[order]
    y_sorted = torch.empty_like(r_sorted)

    for start, end, slot_start, slot_end in runs:
        run_slots_in = slot_in_sorted[start:end]
        if _is_contiguous_in_order(run_slots_in):
            y_sorted[start:end] = _wkv7_fast_one_batch(
                _state_rows_for_sorted_run(state_cache, slot_in_sorted, start, end),
                state_cache[slot_start:slot_end],
                r_sorted[start:end],
                w_sorted[start:end],
                k_sorted[start:end],
                v_sorted[start:end],
                kk_sorted[start:end],
                kka_sorted[start:end],
                positions_sorted[start:end],
            )
            continue
        chunk_start = start
        while chunk_start < end:
            chunk_end = min(chunk_start + _MAX_NONCONTIGUOUS_STATE_GATHER_ROWS, end)
            out_slot_start = slot_start + (chunk_start - start)
            out_slot_end = out_slot_start + (chunk_end - chunk_start)
            y_sorted[chunk_start:chunk_end] = _wkv7_fast_one_batch(
                _state_rows_for_sorted_run(state_cache, slot_in_sorted, chunk_start, chunk_end),
                state_cache[out_slot_start:out_slot_end],
                r_sorted[chunk_start:chunk_end],
                w_sorted[chunk_start:chunk_end],
                k_sorted[chunk_start:chunk_end],
                v_sorted[chunk_start:chunk_end],
                kk_sorted[chunk_start:chunk_end],
                kka_sorted[chunk_start:chunk_end],
                positions_sorted[chunk_start:chunk_end],
            )
            chunk_start = chunk_end

    return y_sorted[inverse]


def _wkv7_seq_batch_inplace_by_slot_runs(
    state_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kk: torch.Tensor,
    kka: torch.Tensor,
    elapsed_t: torch.Tensor,
):
    if _is_contiguous_in_order(slot_mapping):
        slot_start = int(slot_mapping[0].item())
        slot_end = slot_start + slot_mapping.numel()
        return _wkv7_fast_seq_batch(
            state_cache[slot_start:slot_end],
            state_cache[slot_start:slot_end],
            r,
            w,
            k,
            v,
            kk,
            kka,
            elapsed_t,
        )

    order, inverse, runs, has_duplicates = _build_slot_runs(slot_mapping)
    if has_duplicates:
        state = state_cache[slot_mapping].contiguous()
        y = _wkv7_fast_seq_batch(state, state, r, w, k, v, kk, kka, elapsed_t)
        state_cache[slot_mapping] = state
        return y

    r_sorted = r[order]
    w_sorted = w[order]
    k_sorted = k[order]
    v_sorted = v[order]
    kk_sorted = kk[order]
    kka_sorted = kka[order]
    elapsed_sorted = elapsed_t[order]
    y_sorted = torch.empty_like(r_sorted)

    for start, end, slot_start, slot_end in runs:
        y_sorted[start:end] = _wkv7_fast_seq_batch(
            state_cache[slot_start:slot_end],
            state_cache[slot_start:slot_end],
            r_sorted[start:end],
            w_sorted[start:end],
            k_sorted[start:end],
            v_sorted[start:end],
            kk_sorted[start:end],
            kka_sorted[start:end],
            elapsed_sorted[start:end],
        )

    return y_sorted[inverse]


def _wkv7_seq_batch_out_by_slot_runs(
    state_cache: torch.Tensor,
    slot_mapping_in: torch.Tensor,
    slot_mapping_out: torch.Tensor,
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kk: torch.Tensor,
    kka: torch.Tensor,
    elapsed_t: torch.Tensor,
):
    if _is_contiguous_in_order(slot_mapping_in) and _is_contiguous_in_order(slot_mapping_out):
        slot_in_start = int(slot_mapping_in[0].item())
        slot_out_start = int(slot_mapping_out[0].item())
        slot_count = slot_mapping_in.numel()
        return _wkv7_fast_seq_batch(
            state_cache[slot_in_start:slot_in_start + slot_count],
            state_cache[slot_out_start:slot_out_start + slot_count],
            r,
            w,
            k,
            v,
            kk,
            kka,
            elapsed_t,
        )

    order, inverse, runs, has_duplicates = _build_slot_runs(slot_mapping_out)
    if has_duplicates:
        state_in = state_cache[slot_mapping_in].contiguous()
        state_out = state_cache[slot_mapping_out].contiguous()
        y = _wkv7_fast_seq_batch(state_in, state_out, r, w, k, v, kk, kka, elapsed_t)
        state_cache[slot_mapping_out] = state_out
        return y

    slot_in_sorted = slot_mapping_in[order]
    r_sorted = r[order]
    w_sorted = w[order]
    k_sorted = k[order]
    v_sorted = v[order]
    kk_sorted = kk[order]
    kka_sorted = kka[order]
    elapsed_sorted = elapsed_t[order]
    y_sorted = torch.empty_like(r_sorted)

    for start, end, slot_start, slot_end in runs:
        run_slots_in = slot_in_sorted[start:end]
        if _is_contiguous_in_order(run_slots_in):
            y_sorted[start:end] = _wkv7_fast_seq_batch(
                _state_rows_for_sorted_run(state_cache, slot_in_sorted, start, end),
                state_cache[slot_start:slot_end],
                r_sorted[start:end],
                w_sorted[start:end],
                k_sorted[start:end],
                v_sorted[start:end],
                kk_sorted[start:end],
                kka_sorted[start:end],
                elapsed_sorted[start:end],
            )
            continue
        chunk_start = start
        while chunk_start < end:
            chunk_end = min(chunk_start + _MAX_NONCONTIGUOUS_STATE_GATHER_ROWS, end)
            out_slot_start = slot_start + (chunk_start - start)
            out_slot_end = out_slot_start + (chunk_end - chunk_start)
            y_sorted[chunk_start:chunk_end] = _wkv7_fast_seq_batch(
                _state_rows_for_sorted_run(state_cache, slot_in_sorted, chunk_start, chunk_end),
                state_cache[out_slot_start:out_slot_end],
                r_sorted[chunk_start:chunk_end],
                w_sorted[chunk_start:chunk_end],
                k_sorted[chunk_start:chunk_end],
                v_sorted[chunk_start:chunk_end],
                kk_sorted[chunk_start:chunk_end],
                kka_sorted[chunk_start:chunk_end],
                elapsed_sorted[chunk_start:chunk_end],
            )
            chunk_start = chunk_end

    return y_sorted[inverse]
