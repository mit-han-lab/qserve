# original file: https://github.com/vllm-project/vllm/blob/main/vllm/worker/model_runner.py
# modified by: Haotian Tang, Shang Yang, Zhekai Zhang
# @article{lin2024qserve,
#   title={QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving},
#   author={Lin*, Yujun and Tang*, Haotian and Yang*, Shang and Zhang, Zhekai and Xiao, Guangxuan and Gan, Chuang and Han, Song},
#   year={2024}
# }

import os
from typing import Dict, List, Optional, Tuple, Union

import qserve_backend.fused_attention as fused_attention
import torch

from qserve.config import (
    CacheConfig,
    DeviceConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
)
from qserve.logger import init_logger
from qserve.modeling.models.llama_w4a8_unpad import (
    LlamaForCausalLM as LlamaForCausalLMW4A8,
)
from qserve.modeling.models.llama_w8a8_unpad import (
    LlamaForCausalLM as LlamaForCausalLMW8A8,
)
from qserve.modeling.models.llama_w16a16_unpad import (
    LlamaForCausalLM as LlamaForCausalLMW16A16,
)
try:
    from qserve.modeling.models.llava_llama_w4a8_unpad import (
        LlavaLlamaForCausalLM as LlavaLlamaForCausalLMW4A8,
    )
    from qserve.modeling.models.vila_llama_w4a8_unpad import (
        VilaLlamaForCausalLM as VilaLlamaForCausalLMW4A8,
    )
    from qserve.modeling.models.vila_llama_w8a8_unpad import (
        VilaLlamaForCausalLM as VilaLlamaForCausalLMW8A8,
    )
    from qserve.modeling.models.vila_llama_w16a16_unpad import (
        VilaLlamaForCausalLM as VilaLlamaForCausalLMW16A16,
    )
except ImportError:
    print("[Warning] LLAVA and VILA models are not available.")
from qserve.modeling.models.mixtral_w4a8_unpad import (
    MixtralForCausalLM as MixtralForCausalLMW4A8,
)
from qserve.sampling_params import SamplingParams
from qserve.sequence import SamplerOutput, SequenceGroupMetadata
from qserve.utils.input_metadata import InputMetadata
from qserve.utils.utils import STR_DTYPE_TO_TORCH_DTYPE
from qserve.worker.cache_engine import CacheEngine

import qserve.utils.constants

logger = init_logger(__name__)

def tune_llava_patch_embedding(vision_tower, device):
    # run the llava_patch_embedding layer to pre-tune the kernel configuration
    # Without this pre-tuning, the embedding layer can cause significant slowdown due to cuDNN tuning.
    device = vision_tower.device
    patch_embedding = vision_tower.vision_tower.vision_model.embeddings.patch_embedding
    patch_embedding = patch_embedding.to(device)
    image = (
        torch.randn((1, patch_embedding.in_channels, 336, 336))
        .to(device)
        .to(patch_embedding.weight.dtype)
    )
    for i in range(100):
        patch_embedding(image)
    print("Tuned llava_patch_embedding layer.")


def _pad_to_max(x: List[int], max_len: int, pad: int) -> List[int]:
    assert len(x) <= max_len
    return x + [pad] * (max_len - len(x))


def _make_tensor_with_pad(
    x: List[List[int]],
    max_len: int,
    pad: int,
    dtype: torch.dtype,
    device: Optional[Union[str, torch.device]],
) -> torch.Tensor:
    padded_x = [_pad_to_max(x_i, max_len, pad) for x_i in x]
    return torch.tensor(padded_x, dtype=dtype, device=device)


class ModelRunner:
    _sizeof = {torch.float32: 4, torch.float16: 2, torch.bfloat16: 2, torch.int8: 1}

    def __init__(
        self,
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        kv_cache_dtype: torch.dtype,
        is_driver_worker: bool = False,
        precision: str = "w4a8kv4",
        kv_cache_config: Optional[Dict] = None,
        quant_path: Optional[str] = None,
        group_size: int = -1,
        run_vlm: bool = False,
        img_per_seq: int = 0,
        img_rotation: bool = False,
        img_files: str = None,
    ):
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.is_driver_worker = is_driver_worker
        self.kv_cache_config = kv_cache_config
        self.run_vlm = run_vlm
        self.img_per_seq = img_per_seq
        self.img_rotation = img_rotation
        self.img_files = img_files

        # model_config can be None in tests/samplers/test_sampler.py.
        # FIXME(woosuk): This is a hack to make the tests work. Refactor this.
        self.sliding_window = (
            model_config.get_sliding_window() if model_config is not None else None
        )
        self.device_config = (
            device_config if device_config is not None else DeviceConfig()
        )
        self.device = self.device_config.device
        # Note: Shang's important fix here. Otherwise non-GEMM part will run in FP32.
        model_type = model_config.hf_config.architectures[0]

        if (model_type == "LlamaForCausalLM" or model_type == "MistralForCausalLM") and not self.run_vlm:
            if "w4a8" in precision:
                print(f"[INFO] Using {precision} precision")
                self.model = (
                    LlamaForCausalLMW4A8(
                        self.model_config.hf_config,
                        group_size,
                        SamplingParams(
                            temperature=1.0, top_p=1.0, top_k=1, max_tokens=512
                        ),
                        kv_cache_config=self.kv_cache_config,
                        quant_path=quant_path,
                    )
                    .half()
                    .to(self.device)
                )
            elif "w8a8" in precision:
                print(f"[INFO] Using {precision} precision")
                self.model = (
                    LlamaForCausalLMW8A8(
                        self.model_config.hf_config,
                        SamplingParams(
                            temperature=1.0, top_p=1.0, top_k=1, max_tokens=512
                        ),
                        kv_cache_config=self.kv_cache_config,
                        quant_path=quant_path,
                    )
                    .half()
                    .to(self.device)
                )
            elif "w16a16" in precision:
                print(f"[INFO] Using {precision} precision")
                self.model = (
                    LlamaForCausalLMW16A16(
                        self.model_config.hf_config,
                        SamplingParams(
                            temperature=1.0, top_p=1.0, top_k=1, max_tokens=512
                        ),
                        kv_cache_config=self.kv_cache_config,
                        quant_path=quant_path,
                    )
                    .half()
                    .to(self.device)
                )
            else:
                raise ValueError(
                    f"Unsupported model precision: {precision}. Expected w8a8 or w4a8."
                )
        elif model_type == "MixtralForCausalLM" and not self.run_vlm:
            if "w4a8" in precision:
                print(f"[INFO] Using {precision} precision")
                self.model = (
                    MixtralForCausalLMW4A8(
                        self.model_config.hf_config,
                        SamplingParams(
                            temperature=1.0, top_p=1.0, top_k=1, max_tokens=512
                        ),
                        kv_cache_config=self.kv_cache_config,
                        quant_path=quant_path,
                    )
                    .half()
                    .to(self.device)
                )
            else:
                raise ValueError(
                    f"Unsupported model precision: {precision}. Expected w4a8."
                )
        elif (model_type == "VilaLlamaForCausalLM" or model_type == "LlamaForCausalLM") and self.run_vlm:
            assert "kv4" not in precision, "KV4 precision is not allowed for VLM now. Please use higher precision."
            if "w4a8" in precision:
                raise NotImplementedError("W4A8 precision is not allowed for VLM now. Please use higher precision.")
                print(f"[INFO] Using {precision} precision")
                self.model = (
                    VilaLlamaForCausalLMW4A8(
                        self.model_config.vlm_config,
                        group_size,
                        SamplingParams(
                            temperature=1.0, top_p=1.0, top_k=1, max_tokens=512
                        ),
                        kv_cache_config=self.kv_cache_config,
                        quant_path=quant_path,
                        img_rotation=self.img_rotation,
                    )
                    .half()
                    .to(self.device)
                    .eval()
                )
                tune_llava_patch_embedding(self.model.get_vision_tower(), device=self.device)
            elif "w8a8" in precision:
                print(f"[INFO] Using {precision} precision")
                self.model = (
                    VilaLlamaForCausalLMW8A8(
                        self.model_config.vlm_config,
                        SamplingParams(
                            temperature=1.0, top_p=1.0, top_k=1, max_tokens=512
                        ),
                        kv_cache_config=self.kv_cache_config,
                        quant_path=quant_path,
                        img_rotation=self.img_rotation,
                    )
                    .half()
                    .to(self.device)
                    .eval()
                )
                tune_llava_patch_embedding(self.model.get_vision_tower(), device=self.device)
            elif "w16a16" in precision:
                print(f"[INFO] Using {precision} precision")
                self.model = (
                    VilaLlamaForCausalLMW16A16(
                        self.model_config.vlm_config,
                        SamplingParams(
                            temperature=1.0, top_p=1.0, top_k=1, max_tokens=512
                        ),
                        kv_cache_config=self.kv_cache_config,
                        quant_path=quant_path,
                        img_rotation=self.img_rotation,
                    )
                    .half()
                    .to(self.device)
                    .eval()
                )
                tune_llava_patch_embedding(self.model.get_vision_tower(), device=self.device)
            else:
                raise ValueError(
                    f"Unsupported model precision: {precision}. Expected w4a8, w8a8, w16a16kv8."
                )
        else:
            raise ValueError(f"Unsupported model type: {model_type}.")
        self.block_size = None  # Set after initial profiling.

        self.max_context_len_to_capture = (
            self.model_config.max_context_len_to_capture
            if self.model_config is not None
            else 0
        )

        self.kv_cache_dtype = kv_cache_dtype
        self.num_layers = model_config.get_num_layers(parallel_config)

        kv_scale_layer_offsets = (
            torch.arange(
                self.model_config.get_num_layers(self.parallel_config)
            ).unsqueeze(0)
            * self.model_config.max_model_len
            * self.model_config.get_num_kv_heads(self.parallel_config)
        )
        kv_scale_kv_offsets = (
            torch.arange(2).unsqueeze(1)
            * self.model_config.get_num_layers(self.parallel_config)
            * self.model_config.max_model_len
            * self.model_config.get_num_kv_heads(self.parallel_config)
        )
        self.kv_scale_offsets = (kv_scale_layer_offsets + kv_scale_kv_offsets).to(
            self.device_config.device
        )

        free_gpu_memory, total_gpu_memory = torch.cuda.mem_get_info()
        cache_block_mem_size = CacheEngine.get_cache_block_size(
            cache_config.block_size,
            cache_config.cache_bits,
            self.model_config,
            self.parallel_config,
        )
        num_gpu_blocks = int(
            (
                free_gpu_memory
                - total_gpu_memory * (1 - cache_config.gpu_memory_utilization)
            )
            // cache_block_mem_size
        )
        num_cpu_blocks = 10

        manual_num_gpu_blocks = os.environ.get("NUM_GPU_PAGE_BLOCKS")
        if manual_num_gpu_blocks is not None:
            num_gpu_blocks = int(manual_num_gpu_blocks)

        cache_config.num_cpu_blocks = num_cpu_blocks
        cache_config.num_gpu_blocks = num_gpu_blocks
        logger.info(
            f"# GPU blocks: {num_gpu_blocks}, " f"# CPU blocks: {num_cpu_blocks}"
        )
        self.cache_engine = CacheEngine(
            cache_config, model_config, parallel_config, kv_cache_config
        )
        self.cache_events = self.cache_engine.events
        self.gpu_cache = self.cache_engine.gpu_cache
        self.cache_config = cache_config

    def load_model(self) -> None:
        vocab_size = self.model.config.vocab_size

    def set_block_size(self, block_size: int) -> None:
        self.block_size = block_size

        max_num_blocks = (
            self.max_context_len_to_capture + block_size - 1
        ) // block_size

    def _prepare_prompt(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        ifb_mode: bool = True,
    ) -> Tuple[torch.Tensor, InputMetadata,]:
        # kentang-mit@: let's assume that prefix is always none
        assert len(seq_group_metadata_list) > 0
        input_tokens = []
        pil_images = []
        context_lens = []
        block_tables = []
        kv_scales_ptrs = []

        for seq_group_metadata in seq_group_metadata_list:
            assert seq_group_metadata.is_prompt
            seq_ids = list(seq_group_metadata.seq_data.keys())
            assert len(seq_ids) == 1
            seq_id = seq_ids[0]

            seq_data = seq_group_metadata.seq_data[seq_id]
            prompt_tokens = seq_data.get_token_ids()
            input_tokens.append(prompt_tokens)
            context_len = len(prompt_tokens)
            pil_image = seq_data.pil_image
            if pil_image is not None:
                pil_images.append(pil_image)

            if self.run_vlm:
                # Modify token num for img processing
                token_per_img = qserve.utils.constants.LLAVA_DEFAULT_TOKEN_PER_IMAGE
                context_len = context_len + self.img_per_seq * (token_per_img - 1)

            context_lens.append(context_len)

            if seq_group_metadata.block_tables is None:
                # During memory profiling, the block tables are not initialized
                # yet. In this case, we just use a dummy slot mapping.
                continue
            # Compute the slot mapping.
            block_table = seq_group_metadata.block_tables[seq_id]
            block_tables.append(block_table)

        max_prompt_len = max(context_lens)
        input_tokens = torch.cat([torch.tensor(x) for x in input_tokens], dim=0).to(
            device=self.device
        )
        context_lens_tensor = torch.tensor(
            context_lens, dtype=torch.int, device=self.device
        )
        cu_seqlens_tensor = torch.cumsum(context_lens_tensor, dim=0).int()
        cu_seqlens_tensor = torch.nn.functional.pad(cu_seqlens_tensor, (1, 0), value=0)
        # Prepare prefix block tables
        max_prompt_block_table_len = max(len(t) for t in block_tables)
        block_tables = _make_tensor_with_pad(
            block_tables,
            max_len=max_prompt_block_table_len,
            pad=0,
            dtype=torch.long,
            device="cpu",
        )
        layers = self.num_layers
        layer_block_tables = []
        # block_tables_mask = (block_tables != 0).int()
        block_offsets = (
            block_tables
            * self.cache_engine.num_bytes_per_block
            * self._sizeof[STR_DTYPE_TO_TORCH_DTYPE[self.kv_cache_dtype]]
        )

        for l in range(layers):
            base_key_ptrs = self.gpu_cache[l][0].data_ptr()
            base_value_ptrs = self.gpu_cache[l][1].data_ptr()
            # TODO: CHECK offset
            key_ptrs = base_key_ptrs + block_offsets
            value_ptrs = base_value_ptrs + block_offsets
            # key_ptrs *= block_tables_mask
            # value_ptrs *= block_tables_mask
            layer_block_tables.append(
                torch.cat((key_ptrs.unsqueeze(1), value_ptrs.unsqueeze(1)), dim=1).to(
                    self.device
                )
            )
        if self.run_vlm:
            batched_seq_len = input_tokens.size(0) + len(context_lens) * self.img_per_seq * (token_per_img - 1)
        else:
            batched_seq_len = input_tokens.size(0)
        padding_offsets_tensor = fused_attention.compute_padding_offsets(
            cu_seqlens_tensor, max_prompt_len, batched_seq_len 
        )
        # print(padding_offsets_tensor)
        # print(padding_offsets_tensor.shape)
        # exit()

        input_metadata = InputMetadata(
            is_prompt=True,
            context_lens=context_lens_tensor,
            padding_offsets=padding_offsets_tensor,
            cu_seqlens=cu_seqlens_tensor,
            max_seq_len=max_prompt_len,
            max_block_table_len=max_prompt_block_table_len,
            block_tables=layer_block_tables,
            kv_cache_dtype=self.kv_cache_dtype,
            kv_scales=None,
            batched_seq_len=batched_seq_len,
            model=self.model,
            run_vlm=self.run_vlm,
            img_per_seq=self.img_per_seq,
            img_files=self.img_files,
            pil_images=pil_images,
        )
        return (input_tokens, input_metadata)

    def _prepare_decode_ifb(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[torch.Tensor, InputMetadata]:
        assert len(seq_group_metadata_list) > 0
        input_tokens = []
        context_lens = []
        block_tables = []

        for seq_group_metadata in seq_group_metadata_list:
            assert not seq_group_metadata.is_prompt

            seq_ids = list(seq_group_metadata.seq_data.keys())

            for seq_id in seq_ids:
                seq_data = seq_group_metadata.seq_data[seq_id]
                generation_token = seq_data.get_last_token_id()
                input_tokens.append([generation_token])

                seq_len = seq_data.get_len()

                context_len = (
                    seq_len
                    if self.sliding_window is None
                    else min(seq_len, self.sliding_window)
                )
                if self.run_vlm:
                    # Modify token num for img processing
                    token_per_img = qserve.utils.constants.LLAVA_DEFAULT_TOKEN_PER_IMAGE
                    context_len = context_len + self.img_per_seq * (token_per_img - 1)

                context_lens.append(context_len)

                block_table = seq_group_metadata.block_tables[seq_id]

                # if self.sliding_window is not None:
                #     sliding_window_blocks = self.sliding_window // self.block_size
                #     block_table = block_table[-sliding_window_blocks:]
                block_tables.append(block_table)

        max_context_len = max(context_lens)

        input_tokens = _make_tensor_with_pad(
            input_tokens, max_len=1, pad=0, dtype=torch.long, device=self.device
        ).squeeze(1)
        context_lens = torch.tensor(context_lens, dtype=torch.int, device=self.device)
        # kv_scales_tensor = torch.cat(kv_scales_ptrs, dim=2)

        max_block_table_len = max(len(block_table) for block_table in block_tables)
        block_tables = _make_tensor_with_pad(
            block_tables,
            max_len=max_block_table_len,
            pad=0,
            dtype=torch.long,
            device=self.device,
        )
        # block_tables_mask = (block_tables != 0).int()

        layers = self.num_layers

        # Note: padded entries are non-zero now
        block_offsets = (
            block_tables
            * self.cache_engine.num_bytes_per_block
            * self._sizeof[STR_DTYPE_TO_TORCH_DTYPE[self.kv_cache_dtype]]
        )

        base_ptrs = torch.tensor(
            [
                [self.gpu_cache[l][0].data_ptr(), self.gpu_cache[l][1].data_ptr()]
                for l in range(layers)
            ],
            dtype=torch.int64,
            device=self.device,
        )
        # [Layer, 2]
        base_ptrs = (
            base_ptrs.unsqueeze(1)
            .unsqueeze(3)
            .repeat(1, block_offsets.shape[0], 1, block_offsets.shape[1])
        )
        # [Layer, Seq, 2, Len]
        layer_block_tables = base_ptrs + block_offsets.unsqueeze(0).unsqueeze(2).repeat(
            base_ptrs.shape[0], 1, base_ptrs.shape[2], 1
        )
        assert base_ptrs.shape == layer_block_tables.shape

        input_metadata = InputMetadata(
            is_prompt=False,
            cu_seqlens=None,
            padding_offsets=None,
            context_lens=context_lens,
            max_seq_len=max_context_len,
            max_block_table_len=max_block_table_len,
            block_tables=layer_block_tables,
            kv_scales=None,
            kv_cache_dtype=self.kv_cache_dtype,
            batched_seq_len=input_tokens.size(0),
            model=self.model,
            run_vlm=self.run_vlm,
            img_per_seq=self.img_per_seq,
            img_files=self.img_files,
        )

        return (input_tokens, input_metadata)

    def _prepare_decode_no_ifb(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        layer_block_tables: List[torch.Tensor],
        max_block_table_len: int,
        layer_kv_scales: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, InputMetadata, List[int], List[int]]:
        assert len(seq_group_metadata_list) > 0
        input_tokens: List[List[int]] = []
        context_lens: List[int] = []
        assert not seq_group_metadata_list[0].is_prompt

        for seq_group_metadata in seq_group_metadata_list:
            seq_ids = list(seq_group_metadata.seq_data.keys())
            for seq_id in seq_ids:
                seq_data = seq_group_metadata.seq_data[seq_id]
                generation_token = seq_data.get_last_token_id()
                input_tokens.append([generation_token])

        # NOTE: We assume that all sequences have the same length.
        seq_len = seq_group_metadata_list[0].seq_data[0].get_len()
        context_len = (
            seq_len
            if self.sliding_window is None
            else min(seq_len, self.sliding_window)
        )
        context_lens = torch.tensor(
            [
                context_len,
            ]
            * len(seq_group_metadata_list),
            dtype=torch.int,
            device=self.device,
        )
        if self.run_vlm:
            # Modify token num for img processing
            token_per_img = qserve.utils.constants.LLAVA_DEFAULT_TOKEN_PER_IMAGE
            context_len = context_len + self.img_per_seq * (token_per_img - 1)
        max_context_len = context_len

        input_tokens = _make_tensor_with_pad(
            input_tokens, max_len=1, pad=0, dtype=torch.long, device=self.device
        ).squeeze(1)
        input_metadata = InputMetadata(
            is_prompt=False,
            cu_seqlens=None,
            padding_offsets=None,
            context_lens=context_lens,
            max_seq_len=max_context_len,
            max_block_table_len=max_block_table_len,
            block_tables=layer_block_tables,
            kv_scales=layer_kv_scales,
            kv_cache_dtype=self.kv_cache_dtype,
            batched_seq_len=input_tokens.size(0),
            model=self.model,
            run_vlm=self.run_vlm,
            img_per_seq=self.img_per_seq,
            img_files=self.img_files,
        )

        return (input_tokens, input_metadata)

    def prepare_input_tensors(
        self,
        seq_group_metadata_list: Optional[List[SequenceGroupMetadata]],
        ifb_mode: bool = True,
        layer_block_tables: List[torch.Tensor] = None,
        max_block_table_len: int = None,
        layer_kv_scales: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, InputMetadata]:
        # NOTE: We assume that all sequences in the group are all prompts or
        # all decodes.
        is_prompt = seq_group_metadata_list[0].is_prompt
        # Prepare input tensors.
        if is_prompt:
            (input_tokens, input_metadata) = self._prepare_prompt(
                seq_group_metadata_list, ifb_mode
            )
        else:
            if ifb_mode:
                (input_tokens, input_metadata) = self._prepare_decode_ifb(
                    seq_group_metadata_list
                )
            else:
                (input_tokens, input_metadata) = self._prepare_decode_no_ifb(
                    seq_group_metadata_list,
                    layer_block_tables,
                    max_block_table_len,
                    layer_kv_scales=layer_kv_scales,
                )

        return (input_tokens, input_metadata)

    @torch.inference_mode()
    def execute_model(
        self,
        seq_group_metadata_list: Optional[List[SequenceGroupMetadata]],
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
        ifb_mode: bool = True,
        layer_block_tables: List[torch.Tensor] = None,
        max_block_table_len: int = None,
        layer_kv_scales: torch.Tensor = None,
    ) -> Optional[SamplerOutput]:
        (input_tokens, input_metadata) = self.prepare_input_tensors(
            seq_group_metadata_list,
            ifb_mode,
            layer_block_tables,
            max_block_table_len,
            layer_kv_scales=layer_kv_scales,
        )
        model = self.model
        output = model(input_tokens, input_metadata)
        
        if self.run_vlm:
            tokens = model.llm.sample(input_tokens, output, input_metadata)
        else:
            tokens = model.sample(input_tokens, output, input_metadata)
        return tokens
