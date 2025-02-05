# File authors: Haotian Tang, Shang Yang, Yujun Lin, Song Han
# @article{lin2024qserve,
#   title={QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving},
#   author={Lin*, Yujun and Tang*, Haotian and Yang*, Shang and Zhang, Zhekai and Xiao, Guangxuan and Gan, Chuang and Han, Song},
#   year={2024}
# }

import argparse
import time
import gc
import torch

import qserve.utils.constants
from qserve import EngineArgs, LLMEngine, SamplingParams
from qserve.config import ProfilingConfig

max_seq_len = qserve.utils.constants.max_seq_len

import os


def process_requests(
    engine: LLMEngine, batch_size: int, prompt_len: int, generation_len: int
):
    """Continuously process a list of prompts and handle the outputs."""
    request_key = 0
    profiling_config = ProfilingConfig(
        prompt_len=prompt_len, generation_len=generation_len
    )
    for b in range(batch_size):
        engine.add_request(
            str(b),
            prompt=None,
            profiling_config=profiling_config,
            sampling_params=SamplingParams(temperature=0.0),
        )

    if engine.ifb_mode == False:
        # We need to pre-caulcate the block table size for initialization
        block_size = engine.cache_config.block_size
        tot_length = prompt_len + generation_len
        init_num_blocks = (tot_length + block_size - 1) // block_size
        engine.update_init_num_blocks(init_num_blocks)

    # seq_group_metadata_list, scheduler_outputs = engine.step()
    iter = 1

    time_lis = []
    num_tokens = 0
    torch.cuda.synchronize()
    st = time.time()

    while engine.has_unfinished_requests():
        ### Schedule iteration 1 (context stage)
        requests_outputs = engine.step()
        num_tokens += len(requests_outputs)
        # torch.cuda.synchronize()
        if len(requests_outputs) == 0:
            break

        iter += 1
        if engine.profiling_mode and iter == generation_len + 1:
            break
    torch.cuda.synchronize()
    ed = time.time()
    time_lis.append(ed - st)
    return time_lis, num_tokens


def initialize_engine(args: argparse.Namespace) -> LLMEngine:
    """Initialize the LLMEngine from the command line arguments."""
    engine_args = EngineArgs.from_cli_args(args)
    return LLMEngine.from_engine_args(engine_args)


def main(args: argparse.Namespace):
    """Main function that sets up and runs the prompt processing."""

    batch_size = int(os.environ.get("GLOBAL_BATCH_SIZE"))
    prompt_len = 1024
    generation_len = 512
    rounds = 3

    with open("results.csv", "a") as file:
        print("=" * 50, file=file)
        print(
            f"{args.model}: Batch={batch_size}, Input={prompt_len}, Output={generation_len}",
            file=file,
        )

    with torch.no_grad():
        for rnd in range(rounds):
            if rnd < rounds - 1:
                print("[Warmup Round %d]" % rnd)
            engine = initialize_engine(args)
            engine.profiling_mode = True
            # warm up
            time_lis, num_tokens = process_requests(
                engine,
                batch_size=batch_size,
                prompt_len=prompt_len,
                generation_len=generation_len,
            )
            del engine
            torch.cuda.empty_cache()
            gc.collect()

            throughput = num_tokens / sum(time_lis)
            print(f"Round {rnd} Throughput:", throughput, "tokens / second.")
            with open("results.csv", "a") as file:
                print(
                    f"Round {rnd} Throughput:",
                    throughput,
                    "tokens / second.",
                    file=file,
                )

    with open("results.csv", "a") as file:
        print("=" * 50, file=file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Demo on using the LLMEngine class directly"
    )
    parser = EngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    main(args)
