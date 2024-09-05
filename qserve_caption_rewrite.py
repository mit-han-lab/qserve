# File authors: Haotian Tang, Shang Yang, Yujun Lin, Song Han
# @article{lin2024qserve,
#   title={QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving},
#   author={Lin*, Yujun and Tang*, Haotian and Yang*, Shang and Zhang, Zhekai and Xiao, Guangxuan and Gan, Chuang and Han, Song},
#   year={2024}
# }

# CUDA_VISIBLE_DEVICES=0 python qserve_caption_rewrite.py --model /home/haotiant/workspace/checkpoints/Llama-3-8B-Instruct-QServe-W8A8 --ifb-mode   --precision w8a8kv8 --quant-path /home/haotiant/workspace/checkpoints/Llama-3-8B-Instruct-QServe-W8A8 --group-size -1   --max-num-seqs 32 --omit-prompt --max-new-tokens 128 --data_path /home/haotiant/workspace/projects/diffusion_teacher/conceptual/generated_images/cc12m-wds-debug

import argparse
from typing import List, Tuple
import random
import os
import copy

import datasets
import json
import webdataset as wds
from torch.utils.data import DataLoader

from llava.mm_utils import (KeywordsStoppingCriteria, get_model_name_from_path,
                            process_images, tokenizer_image_token)

from tqdm import tqdm

from llava.model import *

import qserve.utils.constants
from qserve import EngineArgs, LLMEngine, SamplingParams
from qserve.conversation import get_conv_template_name, get_conv_template

max_seq_len = qserve.utils.constants.max_seq_len

random.seed(484)

def custom_collate(batch):
    images = [item[0] for item in batch]
    # jsons = [item[1] for item in batch]
    keys = [item[1] for item in batch]
    return images, keys


def create_basic_prompts(conv_t, prompts, max_tokens = 256) -> Tuple[str, SamplingParams]:
    """Create a basic prompt with sampling parameters."""
    sampling_params = SamplingParams(
        temperature=0.7, top_p=1.0, stop_token_ids=[128001, 128009], max_tokens=max_tokens
    )
    ret = []
    for cur_prompt in prompts:
        conv = get_conv_template(conv_t)
        raw_prompt = "Please take the following image caption and attempt to distill it into a single sentence. Remove any redundant lines or descriptions and make it a maximum of 30 words in length."
        raw_prompt += "\nCaption:" + cur_prompt + "\n"
        raw_prompt += "Please only write the caption and no other text.\n"
        # raw_prompt = "<image> Can you describe the image in detail?"
        conv.append_message(conv.roles[0], raw_prompt)
        conv.append_message(conv.roles[1], "")
        prompt = conv.get_prompt()
        ret.append((prompt, sampling_params))
    return ret


def process_requests(engine: LLMEngine, test_prompts: List[Tuple[str, SamplingParams]], pil_images: List = None, keys: List = None):
    """Continuously process a list of prompts and handle the outputs."""
    request_key = 0
    key_id = dict()
    keys = copy.copy(keys)

    while test_prompts or engine.has_unfinished_requests():
        if test_prompts:
            prompt, sampling_params = test_prompts.pop(0)
            if pil_images is not None:
                pil_image = pil_images.pop(0)
                assert keys is not None
                key = keys.pop(0)
                key_id[str(request_key)] = key
            else: 
                pil_image = None
                assert keys is not None
                key = keys.pop(0)
                key_id[str(request_key)] = key
            succeeded = engine.add_request(str(request_key), prompt, sampling_params, pil_image=pil_image)
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
    finished_dict = dict()
    while engine.has_unfinished_requests():
        ### Schedule iteration 1 (context stage)
        requests_outputs = engine.step()
        if len(requests_outputs) == 0:
            break
        for request_output in requests_outputs:
            if request_output["finished"]:
                finished += 1
                finished_dict[key_id[str(request_output['key'])]] = request_output
        iter += 1
        if engine.ifb_mode == False:
            raise NotImplementedError("Non-IFB mode is currently not supported in this script.")
            if iter == max_gen_length:  # Early exit
                for request_output in requests_outputs:
                    print(
                        f"{BG_GREEN}[Conversation {request_output['id']} output]{RESET} {request_output['tokens']}"
                    )
                break
    assert num_test_prompts == finished
    return finished_dict
    


def initialize_engine(args: argparse.Namespace) -> LLMEngine:
    """Initialize the LLMEngine from the command line arguments."""
    engine_args = EngineArgs.from_cli_args(args)
    return LLMEngine.from_engine_args(engine_args)


def main(args: argparse.Namespace):
    """Main function that sets up and runs the prompt processing."""
    engine = initialize_engine(args)
    conversation_template = get_conv_template_name(args.model)

    tar_id = args.job_id * 8 + args.gpu_id
    data_path = args.data_path
    files = sorted(os.listdir(data_path))
    files = list(filter(lambda x: x.endswith(".json"), files))
    json_path = os.path.join(args.data_path, files[tar_id])
    # backup
    os.system(f"cp {json_path} {json_path}.backup")
    if args.info_path is not None:
        with open(args.info_path, "r") as f:
            infos = json.load(f)["shardlist"]
        infos = infos[tar_id]
    else:
        infos = None

    if os.path.exists(json_path):
        print(f'** load from existing json: {json_path} **')
        results = json.load(open(json_path, 'r'))
    else:
        print(f"{json_path} does not exist")
        exit()
    
    # remove json
    model_name = get_model_name_from_path(args.model)
    nsamples = len(results)
    generated_samples = 0
    for key in results:
        if model_name in results[key]:
            generated_samples += 1
    if generated_samples == nsamples:
        print("the entire tar has been captioned, skip")
        exit()
    
    existing_keys = sorted(list(results.keys()))
    for batch_start in tqdm(range(0, nsamples, args.max_num_seqs), desc="Processing batches"):
        st = batch_start
        ed = min(batch_start + args.max_num_seqs, len(existing_keys))
        keys = existing_keys[st : ed]
        prompts = [results[existing_keys[idx]][args.caption_key] for idx in range(st, ed)]
        if all(key in results and model_name in results[key] for key in keys):
            print('already recaption, skip')
            continue
        test_prompts = create_basic_prompts(conv_t=conversation_template, prompts=prompts, max_tokens=args.max_new_tokens)
        outputs = process_requests(engine, test_prompts, keys=keys)

        for key in keys:
            if key not in results:
                results[key] = {}
            output = outputs[key]
            # llama3
            output["text"] = output["text"].replace("<|eot_id|>", "")
            results[key][model_name] = output["text"]

        # Periodically save the results
        with open(json_path, 'w') as file:
            json.dump(results, file)

    # Save the results again after finishing the captioning
    with open(json_path, 'w') as file:
        json.dump(results, file)
    # remove backup
    os.system(f"rm {json_path}.backup")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Demo on using the LLMEngine class directly"
    )
    parser = EngineArgs.add_cli_args(parser)

    parser.add_argument("--job_id", type=int, default=0)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--caption_key", type=str, default="VILA1.5-13b-qserve-w8a8")
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--info_path", type=str, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)

    args = parser.parse_args()
    main(args)
