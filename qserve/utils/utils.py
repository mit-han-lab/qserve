# original file: https://github.com/vllm-project/vllm/blob/main/vllm/utils.py
# modified by: Haotian Tang and Shang Yang
# @article{lin2024qserve,
#   title={QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving},
#   author={Lin*, Yujun and Tang*, Haotian and Yang*, Shang and Zhang, Zhekai and Xiao, Guangxuan and Gan, Chuang and Han, Song},
#   year={2024}
# }

import enum
import os
import socket
import subprocess
from typing import List, TypeVar

import psutil
import torch
from packaging.version import Version, parse

from qserve.logger import init_logger

T = TypeVar("T")
logger = init_logger(__name__)

STR_DTYPE_TO_TORCH_DTYPE = {
    "half": torch.half,
    "bfloat16": torch.bfloat16,
    "float": torch.float,
    "int8": torch.int8,
}


class Device(enum.Enum):
    GPU = enum.auto()
    CPU = enum.auto()


class Counter:
    def __init__(self, start: int = 0) -> None:
        self.counter = start

    def __next__(self) -> int:
        i = self.counter
        self.counter += 1
        return i

    def reset(self) -> None:
        self.counter = 0


def is_hip() -> bool:
    return torch.version.hip is not None


# TODO (kentang-mit@): add this function back in the future
# def get_max_shared_memory_bytes(gpu: int = 0) -> int:
#     """Returns the maximum shared memory per thread block in bytes."""
#     # NOTE: This import statement should be executed lazily since
#     # the Neuron-X backend does not have the `cuda_utils` module.
#     from vllm._C import cuda_utils

#     max_shared_mem = cuda_utils.get_max_shared_memory_per_block_device_attribute(
#         gpu)
#     # value 0 will cause MAX_SEQ_LEN become negative and test_attention.py will fail
#     assert max_shared_mem > 0, "max_shared_mem can not be zero"
#     return int(max_shared_mem)


def get_cpu_memory() -> int:
    """Returns the total CPU memory of the node in bytes."""
    return psutil.virtual_memory().total


def get_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))  # Doesn't need to be reachable
    return s.getsockname()[0]


def get_distributed_init_method(ip: str, port: int) -> str:
    return f"tcp://{ip}:{port}"


def get_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def set_cuda_visible_devices(device_ids: List[int]) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, device_ids))


def get_nvcc_cuda_version() -> Version:
    cuda_home = os.environ.get("CUDA_HOME")
    if not cuda_home:
        cuda_home = "/usr/local/cuda"
        logger.info(
            f"CUDA_HOME is not found in the environment. Using {cuda_home} as CUDA_HOME."
        )
    nvcc_output = subprocess.check_output(
        [cuda_home + "/bin/nvcc", "-V"], universal_newlines=True
    )
    output = nvcc_output.split()
    release_idx = output.index("release") + 1
    nvcc_cuda_version = parse(output[release_idx].split(",")[0])
    return nvcc_cuda_version
