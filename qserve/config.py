# original file: https://github.com/vllm-project/vllm/blob/main/vllm/config.py
# modified by: Haotian Tang and Shang Yang
from typing import Optional, Union

import torch
from transformers import AutoConfig

from qserve.logger import init_logger
from qserve.utils.utils import get_cpu_memory, is_hip

logger = init_logger(__name__)

_GB = 1 << 30


class ModelConfig:
    """Configuration for the model.

    Args:
        model: Name or path of the huggingface model to use.
        tokenizer: Name or path of the huggingface tokenizer to use.
        tokenizer_mode: Tokenizer mode. "auto" will use the fast tokenizer if
            available, and "slow" will always use the slow tokenizer.
        trust_remote_code: Trust remote code (e.g., from HuggingFace) when
            downloading the model and tokenizer.
        download_dir: Directory to download and load the weights, default to the
            default cache directory of huggingface.
        load_format: The format of the model weights to load:
            "auto" will try to load the weights in the safetensors format and
                fall back to the pytorch bin format if safetensors format is
                not available.
            "pt" will load the weights in the pytorch bin format.
            "safetensors" will load the weights in the safetensors format.
            "npcache" will load the weights in pytorch format and store
                a numpy cache to speed up the loading.
            "dummy" will initialize the weights with random values, which is
                mainly for profiling.
        dtype: Data type for model weights and activations. The "auto" option
            will use FP16 precision for FP32 and FP16 models, and BF16 precision
            for BF16 models.
        seed: Random seed for reproducibility.
        revision: The specific model version to use. It can be a branch name,
            a tag name, or a commit id. If unspecified, will use the default
            version.
        code_revision: The specific revision to use for the model code on
            Hugging Face Hub. It can be a branch name, a tag name, or a
            commit id. If unspecified, will use the default version.
        tokenizer_revision: The specific tokenizer version to use. It can be a
            branch name, a tag name, or a commit id. If unspecified, will use
            the default version.
        max_model_len: Maximum length of a sequence (including prompt and
            output). If None, will be derived from the model.
        quantization: Quantization method that was used to quantize the model
            weights. If None, we assume the model weights are not quantized.
        enforce_eager: Whether to enforce eager execution. If True, we will
            disable CUDA graph and always execute the model in eager mode.
            If False, we will use CUDA graph and eager execution in hybrid.
        max_context_len_to_capture: Maximum context len covered by CUDA graphs.
            When a sequence has context length larger than this, we fall back
            to eager mode.
    """

    def __init__(
        self,
        model: str,
        tokenizer: str,
        tokenizer_mode: str,
        trust_remote_code: bool,
        download_dir: Optional[str],
        load_format: str,
        dtype: Union[str, torch.dtype],
        seed: int,
        revision: Optional[str] = None,
        code_revision: Optional[str] = None,
        tokenizer_revision: Optional[str] = None,
        max_model_len: Optional[int] = None,
        quantization: Optional[str] = None,
        enforce_eager: bool = False,
        max_context_len_to_capture: Optional[int] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer_mode = tokenizer_mode
        self.trust_remote_code = trust_remote_code
        self.download_dir = download_dir
        self.load_format = load_format
        self.seed = seed
        self.revision = revision
        self.code_revision = code_revision
        self.tokenizer_revision = tokenizer_revision
        self.quantization = quantization
        self.enforce_eager = enforce_eager
        self.max_context_len_to_capture = max_context_len_to_capture

        self.hf_config = AutoConfig.from_pretrained(
            self.model, trust_remote_code=trust_remote_code
        )
        # TODO (kentang-mit@): make it more general. refer to vllm's _get_and_verify_max_len
        self.dtype = self.hf_config.torch_dtype
        self.max_model_len = self.hf_config.max_position_embeddings
        self._verify_tokenizer_mode()

    def _verify_tokenizer_mode(self) -> None:
        tokenizer_mode = self.tokenizer_mode.lower()
        if tokenizer_mode not in ["auto", "slow"]:
            raise ValueError(
                f"Unknown tokenizer mode: {self.tokenizer_mode}. Must be "
                "either 'auto' or 'slow'."
            )
        self.tokenizer_mode = tokenizer_mode

    def get_sliding_window(self) -> Optional[int]:
        return getattr(self.hf_config, "sliding_window", None)

    def get_vocab_size(self) -> int:
        return self.hf_config.vocab_size

    def get_hidden_size(self) -> int:
        return self.hf_config.hidden_size

    def get_head_size(self) -> int:
        if hasattr(self.hf_config, "head_dim"):
            return self.hf_config.head_dim
        # FIXME(woosuk): This may not be true for all models.
        return self.hf_config.hidden_size // self.hf_config.num_attention_heads

    def get_total_num_kv_heads(self) -> int:
        """Returns the total number of KV heads."""
        # For GPTBigCode & Falcon:
        # NOTE: for falcon, when new_decoder_architecture is True, the
        # multi_query flag is ignored and we use n_head_kv for the number of
        # KV heads.
        falcon_model_types = ["falcon", "RefinedWeb", "RefinedWebModel"]
        new_decoder_arch_falcon = (
            self.hf_config.model_type in falcon_model_types
            and getattr(self.hf_config, "new_decoder_architecture", False)
        )
        if not new_decoder_arch_falcon and getattr(
            self.hf_config, "multi_query", False
        ):
            # Multi-query attention, only one KV head.
            # Currently, tensor parallelism is not supported in this case.
            return 1

        attributes = [
            # For Falcon:
            "n_head_kv",
            "num_kv_heads",
            # For LLaMA-2:
            "num_key_value_heads",
            # For ChatGLM:
            "multi_query_group_num",
        ]
        for attr in attributes:
            num_kv_heads = getattr(self.hf_config, attr, None)
            if num_kv_heads is not None:
                return num_kv_heads

        # For non-grouped-query attention models, the number of KV heads is
        # equal to the number of attention heads.
        return self.hf_config.num_attention_heads

    def get_num_kv_heads(self, parallel_config: "ParallelConfig") -> int:
        """Returns the number of KV heads per GPU."""
        total_num_kv_heads = self.get_total_num_kv_heads()
        # If tensor parallelism is used, we divide the number of KV heads by
        # the tensor parallel size. We will replicate the KV heads in the
        # case where the number of KV heads is smaller than the tensor
        # parallel size so each GPU has at least one KV head.
        return max(1, total_num_kv_heads // parallel_config.tensor_parallel_size)

    def get_num_layers(self, parallel_config: "ParallelConfig") -> int:
        total_num_hidden_layers = self.hf_config.num_hidden_layers
        return total_num_hidden_layers // parallel_config.pipeline_parallel_size


class CacheConfig:
    """Configuration for the KV cache.

    Args:
        block_size: Size of a cache block in number of tokens.
        gpu_memory_utilization: Fraction of GPU memory to use for the
            vLLM execution.
        swap_space: Size of the CPU swap space per GPU (in GiB).
        cache_dtype: Data type for kv cache storage.
    """

    def __init__(
        self,
        block_size: int,
        gpu_memory_utilization: float,
        swap_space: int,
        cache_dtype: str,
        cache_bits: int,
        sliding_window: Optional[int] = None,
    ) -> None:
        self.block_size = block_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.swap_space_bytes = swap_space * _GB
        self.cache_dtype = cache_dtype
        self.cache_bits = cache_bits
        self.sliding_window = sliding_window
        self._verify_args()
        self._verify_cache_dtype()

        # Will be set later.
        self.num_gpu_blocks = None
        self.num_cpu_blocks = None

    def _verify_args(self) -> None:
        if self.gpu_memory_utilization > 1.0:
            raise ValueError(
                "GPU memory utilization must be less than 1.0. Got "
                f"{self.gpu_memory_utilization}."
            )

    def _verify_cache_dtype(self) -> None:
        pass

    def verify_with_parallel_config(
        self,
        parallel_config: "ParallelConfig",
    ) -> None:
        total_cpu_memory = get_cpu_memory()
        # FIXME(woosuk): Here, it is assumed that the GPUs in a tensor parallel
        # group are in the same node. However, the GPUs may span multiple nodes.
        num_gpus_per_node = parallel_config.tensor_parallel_size
        cpu_memory_usage = self.swap_space_bytes * num_gpus_per_node

        msg = (
            f"{cpu_memory_usage / _GB:.2f} GiB out of "
            f"the {total_cpu_memory / _GB:.2f} GiB total CPU memory is "
            "allocated for the swap space."
        )
        if cpu_memory_usage > 0.7 * total_cpu_memory:
            raise ValueError("Too large swap space. " + msg)
        elif cpu_memory_usage > 0.4 * total_cpu_memory:
            logger.warning("Possibly too large swap space. " + msg)


class ParallelConfig:
    """Configuration for the distributed execution.

    Args:
        pipeline_parallel_size: Number of pipeline parallel groups.
        tensor_parallel_size: Number of tensor parallel groups.
        max_parallel_loading_workers: Maximum number of multiple batches
            when load model sequentially. To avoid RAM OOM when using tensor
            parallel and large models.
        disable_custom_all_reduce: Disable the custom all-reduce kernel and
            fall back to NCCL.
    """

    def __init__(
        self,
        pipeline_parallel_size: int,
        tensor_parallel_size: int,
        max_parallel_loading_workers: Optional[int] = None,
        disable_custom_all_reduce: bool = False,
    ) -> None:
        self.pipeline_parallel_size = pipeline_parallel_size
        self.tensor_parallel_size = tensor_parallel_size
        self.max_parallel_loading_workers = max_parallel_loading_workers
        self.disable_custom_all_reduce = disable_custom_all_reduce

        self.world_size = pipeline_parallel_size * tensor_parallel_size
        self._verify_args()

    def _verify_args(self) -> None:
        if self.pipeline_parallel_size > 1:
            raise NotImplementedError("Pipeline parallelism is not supported yet.")
        if not self.disable_custom_all_reduce and self.world_size > 1:
            if is_hip():
                self.disable_custom_all_reduce = True
                logger.info(
                    "Disabled the custom all-reduce kernel because it is not "
                    "supported on AMD GPUs."
                )
            elif self.pipeline_parallel_size > 1:
                self.disable_custom_all_reduce = True
                logger.info(
                    "Disabled the custom all-reduce kernel because it is not "
                    "supported with pipeline parallelism."
                )

        # FIXME(woosuk): Fix the stability issues and re-enable the custom
        # all-reduce kernel.
        if not self.disable_custom_all_reduce and self.world_size > 1:
            self.disable_custom_all_reduce = True
            logger.info(
                "Custom all-reduce kernels are temporarily disabled due to "
                "stability issues. We will re-enable them once the issues are "
                "resolved."
            )


class SchedulerConfig:
    """Scheduler configuration.

    Args:
        max_num_batched_tokens: Maximum number of tokens to be processed in
            a single iteration.
        max_num_seqs: Maximum number of sequences to be processed in a single
            iteration.
        max_model_len: Maximum length of a sequence (including prompt
            and generated text).
        max_paddings: Maximum number of paddings to be added to a batch.
    """

    def __init__(
        self,
        max_num_batched_tokens: Optional[int],
        max_num_seqs: int,
        max_model_len: int,
        max_paddings: int,
    ) -> None:
        if max_num_batched_tokens is not None:
            self.max_num_batched_tokens = max_num_batched_tokens
        else:
            # If max_model_len is too short, use 2048 as the default value for
            # higher throughput.
            self.max_num_batched_tokens = max(max_model_len, 2048)
        self.max_num_seqs = max_num_seqs
        self.max_model_len = max_model_len
        self.max_paddings = max_paddings
        self._verify_args()

    def _verify_args(self) -> None:
        if self.max_num_batched_tokens < self.max_model_len:
            raise ValueError(
                f"max_num_batched_tokens ({self.max_num_batched_tokens}) is "
                f"smaller than max_model_len ({self.max_model_len}). "
                "This effectively limits the maximum sequence length to "
                "max_num_batched_tokens and makes vLLM reject longer "
                "sequences. Please increase max_num_batched_tokens or "
                "decrease max_model_len."
            )
        if self.max_num_batched_tokens < self.max_num_seqs:
            raise ValueError(
                f"max_num_batched_tokens ({self.max_num_batched_tokens}) must "
                "be greater than or equal to max_num_seqs "
                f"({self.max_num_seqs})."
            )


class DeviceConfig:
    def __init__(self, device: str = "cuda") -> None:
        self.device = torch.device(device)


class IFBConfig:
    def __init__(self, ifb_mode: bool = False) -> None:
        self.ifb_mode = ifb_mode


class ProfilingConfig:
    def __init__(self, prompt_len: int, generation_len: int) -> None:
        self.prompt_len = prompt_len
        self.generation_len = generation_len
