import argparse
from typing import List, Tuple

from qserve import EngineArgs, LLMEngine, SamplingParams, RequestOutput
from qserve.modeling.models.llama_w8a8_unpad import LlamaForCausalLM
from qserve.modeling.layers.quantized_linear import (
    W8A8OF16LinearDynamicInputScale,
)
from qserve.modeling.layers.layernorm import RMSNormGeneral
from qserve_backend import fused_kernels
import numpy as np
import torch
from torch import nn

try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_func
    from flash_attn.bert_padding import pad_input
except ImportError:
    print("FlashAttention not found. Install it if you need to train models.")

import transformers
from transformers import AutoConfig, AutoTokenizer

from qserve.utils.input_metadata import InputMetadata
import qserve.utils.constants
import qserve_backend.fused_attention as fused_attention

max_seq_len = qserve.utils.constants.max_seq_len
model_config = AutoConfig.from_pretrained("/data/llm/checkpoints/vicuna-hf/vicuna-7b")


def create_test_prompts() -> List[Tuple[str, SamplingParams]]:
    """Create a list of test prompts with their sampling parameters."""
    return [
        # (
        #     "Beijing is",
        #     SamplingParams(temperature=0.2, top_p=0.7),
        # ),
        (
            "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: Show me some attractions in Boston. ASSISTANT:",
            SamplingParams(temperature=0.7, top_p=0.95, top_k=40),
        ),
        (
            "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: Where is the home court of Golden State Warriors? ASSISTANT:",
            SamplingParams(temperature=0.7, top_p=0.95, top_k=40),
        ),
        (
            "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: Tell me something about NVIDIA. ASSISTANT:",
            SamplingParams(temperature=0.7, top_p=0.95, top_k=40),
        ),
        # (
        #     "Boston is",
        #     SamplingParams(temperature=0.7, top_p=0.95, top_k=40),
        # ),
        # (
        #     "The capital of China is",
        #     SamplingParams(temperature=0.2, top_p=0.7),
        # ),
        # (
        #     "Our teams continue pushing the frontiers of our latest models with safety at the core. They are making rapid progress. In fact, we’re ready to introduce the next generation: Gemini 1.5. It shows dramatic improvements across a number of dimensions and 1.5 Pro achieves comparable quality to 1.0 Ultra, while using less",
        #     SamplingParams(temperature=0.2, top_p=0.7),
        # ),
        # (
        #     "This is an exciting time for AI. New advances in the field have the potential to make AI more helpful for billions of people over the coming years. Since introducing Gemini 1.0, we’ve been testing, refining and enhancing its",
        #     SamplingParams(temperature=0.2, top_p=0.7),
        # ),
        # (
        #     "Write an opening scene for a fantasy fiction novel set in a steampunk 18th century. Be descriptive and historically accurate. Avoid anachronism. The scene should involve Eleanor, a young natural philosopher on a quest to understand the nature of knowledge and learning. Write in the style of Jane Austen. Build mystery throughout the scene and end with an unexpected",
        #     SamplingParams(temperature=0.2, top_p=0.7),
        # ),
    ]


def process_requests(engine: LLMEngine, test_prompts: List[Tuple[str, SamplingParams]]):
    """Continuously process a list of prompts and handle the outputs."""
    request_key = 0

    while test_prompts or engine.has_unfinished_requests():
        if test_prompts:
            prompt, sampling_params = test_prompts.pop(0)
            print(len(engine.tokenizer(prompt)["input_ids"]))
            engine.add_request(str(request_key), prompt, sampling_params)
            request_key += 1

        # request_outputs: List[RequestOutput] = engine.step()

        if not test_prompts:
            break

        # for request_output in request_outputs:
        #     if request_output.finished:
        #         print(request_output)

    input_layernorm = (
        RMSNormGeneral(
            model_config.hidden_size,
            eps=model_config.rms_norm_eps,
            use_per_token_quant=True,
        )
        .cuda()
        .half()
    )
    # model = LlamaForCausalLM(model_config, SamplingParams(temperature=0.7, top_p=0.95, top_k=40)).cuda()
    model = LlamaForCausalLM(
        model_config, SamplingParams(temperature=1.0, top_p=1.0, top_k=1)
    ).cuda()
    model = model.eval()

    # seq_group_metadata_list, scheduler_outputs = engine.step()

    for iter in range(50):
        ### Schedule iteration 1 (context stage)
        seq_group_metadata_list, scheduler_outputs, (input_ids, input_metadata) = (
            engine.step()
        )
        for i, seq_group_metadata in enumerate(seq_group_metadata_list):
            print("Step", iter, "blocks", input_metadata.block_tables[0].shape)
        if iter == 0:
            print("Running context stage...")
        else:
            # if iter == 2:
            #     assert 0
            print("Running decoding stage...")
        output = model(input_ids, input_metadata)
        tokens = model.sample(input_ids, output, input_metadata)
        # print(engine.tokenizer.decode(tokens.reshape(-1).cpu().numpy().tolist()))
        scheduled_seq_groups = scheduler_outputs.scheduled_seq_groups
        for i, seq_group_metadata in enumerate(seq_group_metadata_list):
            scheduled_seq_groups[i].seqs_dict[i].append_token_id(
                tokens[i].item(), {tokens[i].item(): 0}
            )
        # print(
        #     engine.workers[0].gpu_cache[0][0][249],
        #     engine.workers[0].gpu_cache[0][1][249],
        # )

    for i, seq_group_metadata in enumerate(seq_group_metadata_list):
        print(
            "Sequence",
            i,
            ":",
            engine.tokenizer.decode(seq_group_metadata.seq_data[i].get_token_ids()),
        )


def initialize_engine(args: argparse.Namespace) -> LLMEngine:
    """Initialize the LLMEngine from the command line arguments."""
    engine_args = EngineArgs.from_cli_args(args)
    return LLMEngine.from_engine_args(engine_args)


def main(args: argparse.Namespace):
    """Main function that sets up and runs the prompt processing."""
    engine = initialize_engine(args)
    print("DECODED:", engine.tokenizer.decode([12115]))
    test_prompts = create_test_prompts()
    process_requests(engine, test_prompts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Demo on using the LLMEngine class directly"
    )
    parser = EngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    main(args)
