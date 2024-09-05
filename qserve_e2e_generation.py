# File authors: Haotian Tang, Shang Yang, Yujun Lin, Song Han
# @article{lin2024qserve,
#   title={QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving},
#   author={Lin*, Yujun and Tang*, Haotian and Yang*, Shang and Zhang, Zhekai and Xiao, Guangxuan and Gan, Chuang and Han, Song},
#   year={2024}
# }

import argparse
from typing import List, Tuple
import random

import datasets

import qserve.utils.constants
from qserve import EngineArgs, LLMEngine, SamplingParams
from qserve.conversation import get_conv_template_name, get_conv_template

max_seq_len = qserve.utils.constants.max_seq_len
BG_BLUE = "\033[44m"
BG_GREEN = "\033[42m"
BG_PINK = "\033[45m"
RESET = "\033[0m"

random.seed(484)


def create_test_prompts(conv_t, num_prompts=256) -> List[Tuple[str, SamplingParams]]:
    """Create a list of test prompts with their sampling parameters."""
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, stop_token_ids=[128001, 128009], max_tokens=1024)
    dataset = datasets.load_dataset("allenai/WildChat")["train"]
    prompts = []
    i = 0
    len_dataset = len(dataset)
    while len(prompts) < min(len_dataset, num_prompts):
        # prompts.append(dataset[i])
        conv = get_conv_template(conv_t)
        # skip unsafe conversations.
        rand_idx = random.randint(0, len_dataset)
        should_skip = dataset[rand_idx]["toxic"] or dataset[rand_idx]["redacted"]
        if should_skip:
            continue
        raw_prompt = dataset[rand_idx]["conversation"][0]["content"]
        conv.append_message(conv.roles[0], raw_prompt)
        conv.append_message(conv.roles[1], "")
        prompts.append(conv.get_prompt())
    print(f"{BG_PINK}There are {len(prompts)} prompts to be processed.{RESET}")
    return [(prompt, sampling_params) for prompt in prompts]


def process_requests(engine: LLMEngine, test_prompts: List[Tuple[str, SamplingParams]]):
    """Continuously process a list of prompts and handle the outputs."""
    request_key = 0

    while test_prompts or engine.has_unfinished_requests():
        if test_prompts:
            prompt, sampling_params = test_prompts.pop(0)
            succeeded = engine.add_request(str(request_key), prompt, sampling_params)
            if succeeded:
                request_key += 1
        num_test_prompts = request_key

        if not test_prompts:
            break

    if engine.ifb_mode == False:
        # We need to pre-caulcate the block table size for initialization
        block_size = engine.cache_config.block_size
        max_context_length = 128
        max_gen_length = 384
        tot_length = (
            max_context_length + max_gen_length
        )  # Set the upper bound for (prompt + gen) length
        init_num_blocks = (tot_length + block_size - 1) // block_size
        engine.update_init_num_blocks(init_num_blocks)

    # seq_group_metadata_list, scheduler_outputs = engine.step()
    iter = 1
    finished = 0
    while engine.has_unfinished_requests():
        ### Schedule iteration 1 (context stage)
        requests_outputs = engine.step()
        if len(requests_outputs) == 0:
            break
        print(
            BG_BLUE
            + "*" * 5
            + "Iteration %d (remaining req.s = %d)"
            % (iter, len(requests_outputs) + len(engine.scheduler.waiting))
            + "*" * 5
            + RESET
        )
        for request_output in requests_outputs:
            if request_output["finished"]:
                finished += 1
                print(
                    f"{BG_GREEN}[Conversation {request_output['id']} output]{RESET} {request_output['text']}"
                )
        iter += 1
        if engine.ifb_mode == False:
            if iter == max_gen_length:  # Early exit
                for request_output in requests_outputs:
                    print(
                        f"{BG_GREEN}[Conversation {request_output['id']} output]{RESET} {request_output['tokens']}"
                    )
                break
    assert num_test_prompts == finished
    print(f"{BG_PINK}{finished} requests are finished.{RESET}")


def initialize_engine(args: argparse.Namespace) -> LLMEngine:
    """Initialize the LLMEngine from the command line arguments."""
    engine_args = EngineArgs.from_cli_args(args)
    return LLMEngine.from_engine_args(engine_args)


def main(args: argparse.Namespace):
    """Main function that sets up and runs the prompt processing."""
    engine = initialize_engine(args)
    conversation_template = get_conv_template_name(args.model)
    test_prompts = create_test_prompts(
        conv_t=conversation_template, num_prompts=args.max_num_seqs
    )
    process_requests(engine, test_prompts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Demo on using the LLMEngine class directly"
    )
    parser = EngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    main(args)
