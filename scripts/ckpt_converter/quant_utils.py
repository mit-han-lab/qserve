import torch
import torch.nn as nn
from tqdm import tqdm


EMBEDDING_KEYWORDS = ["embed"]
LM_HEAD_KEYWORDS = ["lm_head", "embed_out", "output"]

def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)}


def get_blocks(model):
    if model.__class__.__name__ == "LlamaForCausalLM":
        layers = model.model.layers
    elif model.__class__.__name__ == "LlavaLlamaForCausalLM":
        # layers = [model.model.layers, model.model.vision_tower.vision_tower.vision_model.encoder.layers]
        layers = model.model.layers
    elif "mpt" in str(model.__class__).lower():
        layers = model.transformer.blocks
    elif "falcon" in str(model.__class__).lower():
        layers = model.transformer.h
    elif "bigcode" in str(model.__class__).lower():
        layers = model.transformer.h
    elif "neox" in str(model.__class__).lower():
        layers = model.gpt_neox.layers
    elif "mistral" in str(model.__class__).lower():
        layers = model.model.layers
    else:
        raise NotImplementedError(type(model))
    return layers


def set_op_by_name(layer, name, new_module):
    levels = name.split(".")
    if len(levels) > 1:
        mod_ = layer
        for l_idx in range(len(levels) - 1):
            if levels[l_idx].isdigit():
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], new_module)
    else:
        setattr(layer, name, new_module)


class ScaledActivation(nn.Module):
    def __init__(self, module, scales):
        super().__init__()
        self.act = module
        self.scales = nn.Parameter(scales.data)

    def forward(self, x):
        return self.act(x) / self.scales.view(1, 1, -1).to(x.device)

def scale_activations(module):
    param = next(module.parameters())
    dtype = param.dtype
    device = param.device
    if "mptblock" in str(module.__class__.__name__).lower():
        if isinstance(module.ffn.act, ScaledActivation):
            return
        c = module.ffn.up_proj.out_features
        act = ScaledActivation(
            module.ffn.act, torch.ones(c, dtype=dtype, device=device)
        )
        set_op_by_name(module, "ffn.act", act)
    elif "falcon" in str(module.__class__).lower():
        if isinstance(module.mlp.act, ScaledActivation):
            return
        c = module.mlp.dense_h_to_4h.out_features
        act = ScaledActivation(
            module.mlp.act, torch.ones(c, dtype=dtype, device=device)
        )
        set_op_by_name(module, "mlp.act", act)
    elif "bigcode" in str(module.__class__).lower():
        if isinstance(module.mlp.act, ScaledActivation):
            return
        c = module.mlp.c_proj.out_features
        act = ScaledActivation(
            module.mlp.act, torch.ones(c, dtype=dtype, device=device)
        )
        set_op_by_name(module, "mlp.act", act)
    elif "neox" in str(module.__class__).lower():
        if isinstance(module.mlp.act, ScaledActivation):
            return
        c = module.mlp.dense_h_to_4h.out_features
        act = ScaledActivation(
            module.mlp.act, torch.ones(c, dtype=dtype, device=device)
        )
        set_op_by_name(module, "mlp.act", act)


# core quantization method (simulated quantization)
def pseudo_quantize_tensor(
    w, n_bit=8, zero_point=True, q_group_size=-1, inplace=False, get_scale_zp=False
):
    org_w_shape = w.shape
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size)
    assert w.dim() == 2
    if zero_point:
        max_val = w.amax(dim=1, keepdim=True)
        min_val = w.amin(dim=1, keepdim=True)
        max_int = 2**n_bit - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
    else:  # we actually never used this
        assert min_val is None
        max_val = w.abs().amax(dim=1, keepdim=True)
        max_val = max_val.clamp(min=1e-5)
        max_int = 2 ** (n_bit - 1) - 1
        min_int = -(2 ** (n_bit - 1))
        scales = max_val / max_int
        zeros = 0

    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(w).sum() == 0

    if inplace:
        (
            (w.div_(scales).round_().add_(zeros)).clamp_(min_int, max_int).sub_(zeros)
        ).mul_(scales)
    else:
        w = (
            torch.clamp(torch.round(w / scales) + zeros, min_int, max_int) - zeros
        ) * scales
    assert torch.isnan(w).sum() == 0

    w = w.reshape(org_w_shape)

    if get_scale_zp:
        return w, scales.view(w.shape[0], -1), zeros.view(w.shape[0], -1)
    else:
        return w


@torch.no_grad()
def pseudo_quantize_model_weight(
    model,
    w_bit,
    q_config,
):

    layers = get_blocks(model)
    for i in tqdm(range(len(layers)), desc="pseudo weight quantization..."):
        named_linears = get_named_linears(layers[i])
        for n, m in named_linears.items():
            m.cuda()
            m.weight.data = pseudo_quantize_tensor(
                m.weight.data, n_bit=w_bit, **q_config
            )
            m.cpu()