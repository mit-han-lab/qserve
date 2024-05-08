# This file is transfered from: https://github.com/vllm-project/vllm

"""Utilities for downloading and initializing model weights."""
import glob
import json
import os
from typing import Any, Iterator, List, Optional, Tuple

import filelock
import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import safe_open
from tqdm.auto import tqdm


class Disabledtqdm(tqdm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, disable=True)


def get_lock(model_name_or_path: str, cache_dir: Optional[str] = None):
    lock_dir = cache_dir if cache_dir is not None else "/tmp"
    lock_file_name = model_name_or_path.replace("/", "-") + ".lock"
    lock = filelock.FileLock(os.path.join(lock_dir, lock_file_name))
    return lock


def prepare_hf_model_weights(
    model_name_or_path: str,
    cache_dir: Optional[str] = None,
    use_safetensors: bool = False,
    fall_back_to_pt: bool = True,
    revision: Optional[str] = None,
) -> Tuple[str, List[str], bool]:
    # Download model weights from huggingface.
    is_local = os.path.isdir(model_name_or_path)
    if use_safetensors:
        allow_patterns = ["*.safetensors"]
    else:
        # Some quantized models use .pt files for storing the weights.
        allow_patterns = ["*.bin", "*.pt"]
    if not is_local:
        # Use file lock to prevent multiple processes from
        # downloading the same model weights at the same time.
        with get_lock(model_name_or_path, cache_dir):
            hf_folder = snapshot_download(
                model_name_or_path,
                allow_patterns=allow_patterns,
                cache_dir=cache_dir,
                tqdm_class=Disabledtqdm,
                revision=revision,
            )
    else:
        hf_folder = model_name_or_path
    hf_weights_files: List[str] = []
    for pattern in allow_patterns:
        hf_weights_files += glob.glob(os.path.join(hf_folder, pattern))
    if not use_safetensors:
        # Exclude files that are not needed for inference.
        # https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/trainer.py#L227-L233
        blacklist = [
            "training_args.bin",
            "optimizer.bin",
            "optimizer.pt",
            "scheduler.pt",
            "scaler.pt",
        ]
        hf_weights_files = [
            f for f in hf_weights_files if not any(f.endswith(x) for x in blacklist)
        ]

    if len(hf_weights_files) == 0 and use_safetensors and fall_back_to_pt:
        return prepare_hf_model_weights(
            model_name_or_path,
            cache_dir=cache_dir,
            use_safetensors=False,
            fall_back_to_pt=False,
            revision=revision,
        )

    if len(hf_weights_files) == 0:
        raise RuntimeError(f"Cannot find any model weights with `{model_name_or_path}`")

    return hf_folder, hf_weights_files, use_safetensors


def hf_model_weights_iterator(
    model_name_or_path: str,
    cache_dir: Optional[str] = None,
    load_format: str = "auto",
    revision: Optional[str] = None,
) -> Iterator[Tuple[str, torch.Tensor]]:
    use_safetensors = False
    use_np_cache = False
    fall_back_to_pt = False
    if load_format == "auto":
        use_safetensors = True
        fall_back_to_pt = True
    elif load_format == "safetensors":
        use_safetensors = True
    elif load_format == "pt":
        pass
    elif load_format == "npcache":
        use_np_cache = True
    else:
        raise ValueError(f"Unknown load_format: {load_format}")

    hf_folder, hf_weights_files, use_safetensors = prepare_hf_model_weights(
        model_name_or_path,
        cache_dir=cache_dir,
        use_safetensors=use_safetensors,
        fall_back_to_pt=fall_back_to_pt,
        revision=revision,
    )

    if use_np_cache:
        # Currently np_cache only support *.bin checkpoints
        assert use_safetensors is False

        # Convert the model weights from torch tensors to numpy arrays for
        # faster loading.
        np_folder = os.path.join(hf_folder, "np")
        os.makedirs(np_folder, exist_ok=True)
        weight_names_file = os.path.join(np_folder, "weight_names.json")
        # Use file lock to prevent multiple processes from
        # dumping the same model weights to numpy at the same time.
        with get_lock(model_name_or_path, cache_dir):
            if not os.path.exists(weight_names_file):
                weight_names = []
                for bin_file in hf_weights_files:
                    state = torch.load(bin_file, map_location="cpu")
                    for name, param in state.items():
                        param_path = os.path.join(np_folder, name)
                        with open(param_path, "wb") as f:
                            np.save(f, param.cpu().detach().numpy())
                        weight_names.append(name)
                with open(weight_names_file, "w") as f:
                    json.dump(weight_names, f)

        with open(weight_names_file, "r") as f:
            weight_names = json.load(f)

        for name in weight_names:
            param_path = os.path.join(np_folder, name)
            with open(param_path, "rb") as f:
                param = np.load(f)
            yield name, torch.from_numpy(param)
    elif use_safetensors:
        for st_file in hf_weights_files:
            with safe_open(st_file, framework="pt") as f:
                for name in f.keys():
                    param = f.get_slice(name)
                    yield name, param
    else:
        for bin_file in hf_weights_files:
            state = torch.load(bin_file, map_location="cpu")
            for name, param in state.items():
                yield name, param
            del state
            # torch.cuda.empty_cache()


def convert_pyslice_to_tensor(x: Any) -> torch.Tensor:
    """convert PySafeSlice object from safetensors to torch.Tensor

    PySafeSlice object supports indexing, which is done before loading the
    actual tensor and can reduce the amount of memory being read into the
    memory. However, it does not support more advanced functionalities
    like `.view()` or `.t()`. Therefore, if we need to modify the loaded
    tensor with these more complicated operators, we need to convert to
    tensor first.
    """
    if not isinstance(x, torch.Tensor):
        x = x[:]
    return x


def load_padded_tensor_parallel_vocab(
    param: torch.Tensor,
    loaded_weight: Any,  # `torch.Tensor` or `PySafeSlice`
    tensor_model_parallel_rank: int,
) -> None:
    shard_size = param.shape[0]
    start_idx = tensor_model_parallel_rank * shard_size
    end_idx = (tensor_model_parallel_rank + 1) * shard_size
    loaded_weight = loaded_weight[start_idx:end_idx]
    loaded_weight = convert_pyslice_to_tensor(loaded_weight)
    param[: loaded_weight.shape[0]].copy_(loaded_weight)


def load_tensor_parallel_weights(
    param: torch.Tensor,
    loaded_weight: Any,  # `torch.Tensor` or `PySafeSlice`
    param_name: str,
    column_parallel_weight_names: List[str],
    row_parallel_weight_names: List[str],
    tensor_model_parallel_rank: int,
) -> None:
    for p in column_parallel_weight_names:
        if p in param_name:
            shard_size = param.shape[0]
            start_idx = tensor_model_parallel_rank * shard_size
            end_idx = (tensor_model_parallel_rank + 1) * shard_size
            loaded_weight = loaded_weight[start_idx:end_idx]
            break
    for p in row_parallel_weight_names:
        if p in param_name:
            shard_size = param.shape[1]
            start_idx = tensor_model_parallel_rank * shard_size
            end_idx = (tensor_model_parallel_rank + 1) * shard_size
            loaded_weight = loaded_weight[:, start_idx:end_idx]
            break

    loaded_weight = convert_pyslice_to_tensor(loaded_weight)
    assert param.shape == loaded_weight.shape, (
        f"{param_name} shape mismatch between model and checkpoint: "
        f"{param.shape} != {loaded_weight.shape}"
    )
    param.data.copy_(loaded_weight)
