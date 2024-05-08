import os
import torch
import argparse
from qserve.modeling.layers.quantized_linear import W4A8OF16LinearDynamicInputScale
from quant_utils import get_blocks, get_named_linears, scale_activations, set_op_by_name
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, modeling_utils
from tqdm import tqdm

def skip(*args, **kwargs):
    pass

torch.nn.init.kaiming_uniform_ = skip
torch.nn.init.kaiming_normal_ = skip
torch.nn.init.uniform_ = skip
torch.nn.init.normal_ = skip


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-type", type=str, default="LLaMa", help="type of the model"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="/data/llama-2-7b-hf/",
        help="path to the hf model configs",
    )
    parser.add_argument(
        "--quant-path",
        type=str,
        default="/data/llama-2-7b-hf-fake-quant/",
        help="path to the fake quantized model dumped by LMQuant, including model.pt and scale.pt",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=-1,
        help="group size for quantization, -1 means per-channel quant",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="../../qserve_checkpoints/",
        help="path to save the real quantized model checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )

    args = parser.parse_args()
    assert args.model_type.lower() in [
        "llama",
    ], "We only support llama architecture for now."

    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, use_fast=False, trust_remote_code=True
    )

    modeling_utils._init_weights = False
    model = AutoModelForCausalLM.from_config(config)

    fake_quant_ckpt = torch.load(args.quant_path+'/model.pt', map_location=args.device)
    quant_params = torch.load(args.quant_path+'/scale.pt', map_location=args.device)

    w_bit = 4
    q_config = {"zero_point": True, "q_group_size": args.group_size}

    layers = get_blocks(model)
    for i in tqdm(
        range(len(layers)),
    ):
        layer = layers[i]
        named_linears = get_named_linears(layer)
        scale_activations(layer)

        for name, module in named_linears.items():
            layer_weight_name = f"model.layers.{i}.{name}"
            s1_scale = quant_params[f"{layer_weight_name}.weight.scale.0"]
            if q_config["q_group_size"] != -1:
                assert f"{layer_weight_name}.weight.scale.1" in quant_params.keys(), f"{layer_weight_name}.weight.scale.1 not found in quant_params. Please check if you are using per-group quantization."
                s2_scale = quant_params[f"{layer_weight_name}.weight.scale.1"]
            else:
                assert f"{layer_weight_name}.weight.scale.1" not in quant_params.keys(), f"{layer_weight_name}.weight.scale.1 found in quant_params. Please check if you are using per-channel quantization."
                s2_scale = None
            zeros = quant_params[f"{layer_weight_name}.weight.zero"].to(torch.int8)
            if zeros.min() < 0:
                zeros = zeros + 8
            module.weight.data = fake_quant_ckpt[f"{layer_weight_name}.weight"]
            module = module.cpu()
            
            q_linear = W4A8OF16LinearDynamicInputScale.from_linear(
                module, w_bit, q_config["q_group_size"], init_only=False, s1_scale=s1_scale, s2_scale=s2_scale, zeros=zeros
            )
            # q_linear.to(next(layer.parameters()).device)
            set_op_by_name(layer, name, q_linear)
    
    # Sync layers other than gemm layers
    for key, param in model.named_parameters():
        if "_proj" in key:
            pass
        else:
            param.data = fake_quant_ckpt[key].data

    torch.save(model.state_dict(), f"{args.output_path}/quant_model.pt")

    # Organize checkpoint and config files
    model_name = args.model_path.rstrip("/").split("/")[-1]
    model_name = model_name + "-w4a8-per-channel" if args.group_size == -1 else model_name + f"-w4a8-g{args.group_size}"
    os.system(f"mkdir -p {args.output_path}/{model_name}")
    os.system(f"mv {args.output_path}/quant_model.pt {args.output_path}/{model_name}/pytorch_model.bin")
    os.system(f"scp {args.model_path}/*.json {args.output_path}/{model_name}")
    os.system(f"scp {args.model_path}/tokenizer.model {args.output_path}/{model_name}")
    os.system(f"rm -f {args.output_path}/{model_name}/pytorch_model.bin.index.json")

