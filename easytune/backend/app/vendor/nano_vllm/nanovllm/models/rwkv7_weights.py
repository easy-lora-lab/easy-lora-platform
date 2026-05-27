import gc

import torch
import torch.nn.functional as F

from nanovllm.models import rwkv7_ops


def _rwkv7_weight_device() -> torch.device:
    if torch.cuda.is_available() and torch.cuda.current_device() >= 0:
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return torch.device("cpu")


def _should_transpose_weight(key: str) -> bool:
    if rwkv7_ops.is_lowrank_weight(key):
        return True
    return (
        ("key.weight" in key and not rwkv7_ops.is_orig_linear_weight(key))
        or ("value.weight" in key and not rwkv7_ops.is_orig_linear_weight(key))
        or ("receptance.weight" in key and not rwkv7_ops.is_orig_linear_weight(key))
        or ("output.weight" in key and not rwkv7_ops.is_orig_linear_weight(key))
        or ("head.weight" in key and not rwkv7_ops.is_orig_linear_weight(key))
    )


def prepare_rwkv7_weight_state(z: dict[str, torch.Tensor], hidden_size: int) -> dict[str, torch.Tensor]:
    z = dict(z)
    keys = list(z.keys())
    dtype = torch.get_default_dtype()
    device = _rwkv7_weight_device()

    for key in keys:
        value = z[key].squeeze()
        if _should_transpose_weight(key):
            value = value.t()
        z[key] = value.to(dtype=dtype, device=device)

        if key.endswith("att.r_k"):
            z[key] = z[key].flatten()

        z[key] = z[key].contiguous()
        if rwkv7_ops.is_lowrank_weight(key):
            if rwkv7_ops.LOWRANK_WEIGHT in ("transpose", "both"):
                z[key + ".t"] = value.t().to(dtype=dtype, device=device).contiguous()

    if "emb.weight" in z and "blocks.0.ln0.weight" in z and "blocks.0.ln0.bias" in z:
        z["emb.weight"] = F.layer_norm(
            z["emb.weight"],
            (hidden_size,),
            weight=z["blocks.0.ln0.weight"],
            bias=z["blocks.0.ln0.bias"],
        )

    if "blocks.0.att.a0" in z:
        z["blocks.0.att.v0"] = z["blocks.0.att.a0"]
    if "blocks.0.att.a1" in z:
        z["blocks.0.att.v1"] = z["blocks.0.att.a1"]
    if "blocks.0.att.a2" in z:
        z["blocks.0.att.v2"] = z["blocks.0.att.a2"]

    if rwkv7_ops.RKV_MODE != "off" and not rwkv7_ops.use_orig_linear("att_c2c"):
        layers = sorted({
            int(key.split(".")[1])
            for key in z
            if key.startswith("blocks.") and key.endswith(".att.receptance.weight")
        })
        for layer in layers:
            prefix = f"blocks.{layer}.att."
            rkv_keys = (prefix + "receptance.weight", prefix + "key.weight", prefix + "value.weight")
            if all(key in z for key in rkv_keys):
                z[prefix + "rkv.weight"] = torch.stack(tuple(z[key] for key in rkv_keys)).contiguous()
    return z


def load_rwkv7_pth_into_model(model, pth_path: str):
    z = prepare_rwkv7_weight_state(torch.load(pth_path, map_location="cpu"), hidden_size=model.hidden_size)

    model.z = z
    load_rwkv7_state_into_modules(model)


def load_rwkv7_state_into_modules(model):
    z = model.z

    if model.emb.tp_size == 1:
        model.emb.weight.data = z["emb.weight"]
    else:
        model.emb.weight.data.copy_(z["emb.weight"])

    for idx, block in enumerate(model.blocks):
        bbb = f"blocks.{idx}."
        att = f"blocks.{idx}.att."
        ffn = f"blocks.{idx}.ffn."

        block.ln1.gamma.data = z[bbb + "ln1.weight"]
        block.ln1.beta.data = z[bbb + "ln1.bias"]
        block.ln2.gamma.data = z[bbb + "ln2.weight"]
        block.ln2.beta.data = z[bbb + "ln2.bias"]

        block.ffn.key_proj.weight.data = z[ffn + "key.weight"]
        block.ffn.value_proj.weight.data = z[ffn + "value.weight"]
        block.ffn.register_buffer("x_k", z[ffn + "x_k"])

        block.att.register_buffer("x_r", z[att + "x_r"])
        block.att.register_buffer("x_w", z[att + "x_w"])
        block.att.register_buffer("x_k", z[att + "x_k"])
        block.att.register_buffer("x_v", z[att + "x_v"])
        block.att.register_buffer("x_a", z[att + "x_a"])
        block.att.register_buffer("x_g", z[att + "x_g"])
        block.att.register_buffer("w0", z[att + "w0"])
        block.att.w1_proj.weight.data = z[att + "w1"]
        block.att.register_buffer("w1_t", z.get(att + "w1.t"))
        block.att.w2_proj.weight.data = z[att + "w2"]
        block.att.register_buffer("w2_t", z.get(att + "w2.t"))
        block.att.register_buffer("a0", z[att + "a0"])
        block.att.a1_proj.weight.data = z[att + "a1"]
        block.att.register_buffer("a1_t", z.get(att + "a1.t"))
        block.att.a2_proj.weight.data = z[att + "a2"]
        block.att.register_buffer("a2_t", z.get(att + "a2.t"))
        block.att.register_buffer("v0", z[att + "v0"])
        block.att.v1_proj.weight.data = z[att + "v1"]
        block.att.register_buffer("v1_t", z.get(att + "v1.t"))
        block.att.v2_proj.weight.data = z[att + "v2"]
        block.att.register_buffer("v2_t", z.get(att + "v2.t"))
        block.att.g1_proj.weight.data = z[att + "g1"]
        block.att.register_buffer("g1_t", z.get(att + "g1.t"))
        block.att.g2_proj.weight.data = z[att + "g2"]
        block.att.register_buffer("g2_t", z.get(att + "g2.t"))
        block.att.register_buffer("k_k", z[att + "k_k"].squeeze())
        block.att.register_buffer("k_a", z[att + "k_a"].squeeze())
        block.att.register_buffer("r_k", z[att + "r_k"].flatten())
        block.att.receptance_proj.weight.data = z[att + "receptance.weight"]
        block.att.key_proj.weight.data = z[att + "key.weight"]
        block.att.value_proj.weight.data = z[att + "value.weight"]
        block.att.output_proj.weight.data = z[att + "output.weight"]
        block.att.rkv_weight = z.get(att + "rkv.weight")
        block.att.register_buffer("ln_x_weight", z[att + "ln_x.weight"])
        block.att.register_buffer("ln_x_bias", z[att + "ln_x.bias"])

    model.ln_out.gamma.data = z["ln_out.weight"]
    model.ln_out.beta.data = z["ln_out.bias"]


def apply_rwkv7_post_load_quantization(causal_lm):
    if getattr(causal_lm.config, "rwkv_quant_int8", False):
        raise RuntimeError("RWKV7 faster3a inference requires fp16 attention/FFN weights; rwkv_quant_int8 is unsupported.")

    if causal_lm.lm_head.tp_size == 1:
        causal_lm.lm_head.weight.data = causal_lm.model.z["head.weight"].t()
    else:
        shard_size = causal_lm.lm_head.num_embeddings_per_partition
        start_idx = causal_lm.lm_head.tp_rank * shard_size
        causal_lm.lm_head.weight.data.copy_(
            causal_lm.model.z["head.weight"].narrow(0, start_idx, shard_size).t().contiguous()
        )
    if "head.weight" in causal_lm.model.z:
        del causal_lm.model.z["head.weight"]

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
