# File authors: Haotian Tang, Shang Yang, Yujun Lin, Song Han
# @article{lin2024qserve,
#   title={QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving},
#   author={Lin*, Yujun and Tang*, Haotian and Yang*, Shang and Zhang, Zhekai and Xiao, Guangxuan and Gan, Chuang and Han, Song},
#   year={2024}
# }

from typing import Optional

import torch
from xformers.ops import AttentionBias


class ActivationBuffer:
    """
    Pre-allocated Buffer for activation in the model.

    Args:
        model: The input model
        batched_seq_len: The batched sequence length. Sum of all the sequence lengths in the batch.
    """

    def __init__(self, model, batched_seq_len: int):
        self.model_class = model.__class__.__name__
        self.model_dtype = model.model.embed_tokens.weight.dtype
        self.device = model.model.embed_tokens.weight.device
        assert self.model_class in [
            "LlamaForCausalLM",
            "MixtralForCausalLM",
        ], f"model_class: {self.model_class} is currently not supported."
        assert (
            self.model_dtype == torch.float16
        ), f"model_dtype is expected to be fp16. Current: {self.model_dtype}."

        self.batched_seq_len = batched_seq_len
        assert (
            self.batched_seq_len > 0
        ), f"batched_seq_len is expected to be greater than 0 to allocate activation buffer. Current: {self.batched_seq_len}."

        self.q_size = model.q_size
        self.kv_size = model.kv_size
        self.intermediate_size = model.config.intermediate_size
        self.hidden_size = model.config.hidden_size

    def allocate_activation_buffer(self):
        if self.model_class == "LlamaForCausalLM":
            self.__allocate_activation_buffer_llama()
        elif self.model_class == "MixtralForCausalLM":
            raise NotImplementedError("MixtralForCausalLM is not supported yet.")
        else:
            raise NotImplementedError(
                f"model_class: {self.model_class} is currently not supported."
            )

    def __allocate_activation_buffer_llama(self):
        # Allocate fp16 activation buffer.
        self.act_buffer = torch.empty(
            (
                self.batched_seq_len
                * max(self.q_size + 2 * self.kv_size, 2 * self.intermediate_size)
            ),
            device=self.device,
            dtype=torch.float16,
        )
        self.qkv_proj_act_buffer = self.act_buffer[
            : self.batched_seq_len * (self.q_size + 2 * self.kv_size)
        ].view(self.batched_seq_len, self.q_size + 2 * self.kv_size)
        self.out_down_proj_act_buffer = self.act_buffer[
            : self.batched_seq_len * self.hidden_size
        ].view(self.batched_seq_len, self.hidden_size)
        self.gate_up_proj_act_buffer = self.act_buffer[
            : self.batched_seq_len * 2 * self.intermediate_size
        ].view(self.batched_seq_len, 2 * self.intermediate_size)

        # Allocate quantized activation buffer.
        self.quantized_act_buffer = torch.empty(
            (self.batched_seq_len * max(self.hidden_size, self.intermediate_size)),
            device=self.device,
            dtype=torch.int8,
        )
        self.quantized_hidden_states_buffer = self.quantized_act_buffer[
            : self.batched_seq_len * self.hidden_size
        ].view(self.batched_seq_len, self.hidden_size)
        self.quantized_mlp_act_buffer = self.quantized_act_buffer[
            : self.batched_seq_len * self.intermediate_size
        ].view(self.batched_seq_len, self.intermediate_size)

        self.quantized_scale_buffer = torch.empty(
            (self.batched_seq_len), device=self.device, dtype=torch.float16
        )
        self.quantized_sum_buffer = torch.empty(
            (self.batched_seq_len), device=self.device, dtype=torch.float16
        )


class InputMetadata:
    """Metadata for input sequences. Used for PagedAttention.

    Args:
        seq_groups: List of (seq_ids, sampling_params).
        seq_data: Seq_id -> SequenceData.
        prompt_lens: Lengths of prompts.
        slot_mapping: The address to write the new KV to of each token.
        context_lens: the length of attention context for each generation token.
        max_context_len: The maximum context length.
        block_tables: The block tables. (Seq id -> list of physical block)
    """

    def __init__(
        self,
        # seq_groups: List[Tuple[List[int], SamplingParams]],
        # seq_data: Dict[int, SequenceData],
        is_prompt: bool,
        padding_offsets: torch.Tensor,
        cu_seqlens: torch.Tensor,
        context_lens: torch.Tensor,
        # slot_mapping: torch.Tensor,
        # context_lens: torch.Tensor,
        max_seq_len: int,
        block_tables: torch.Tensor,
        max_block_table_len: int,
        # selected_token_indices: torch.Tensor,
        # categorized_sample_indices: Dict[SamplingType, torch.Tensor],
        # sliding_window: Optional[int] = None,
        # start_pos: int,
        kv_scales: torch.Tensor,
        kv_cache_dtype: torch.dtype,
        batched_seq_len: int,
        model: torch.nn.Module,
    ) -> None:
        # self.seq_groups = seq_groups
        # self.seq_data =
        self.is_prompt = is_prompt
        self.padding_offsets = padding_offsets
        self.cu_seqlens = cu_seqlens
        self.context_lens = context_lens
        # self.slot_mapping = slot_mapping
        self.max_seq_len = max_seq_len
        self.block_tables = block_tables
        self.max_block_table_len = max_block_table_len
        self.kv_scales = kv_scales
        # self.selected_token_indices = selected_token_indices
        # self.categorized_sample_indices = categorized_sample_indices

        # self.max_prompt_len = max(prompt_lens) if prompt_lens else 0
        # self.to_cache = None
        # if sliding_window is not None:
        #     # We need to keep the positions of sliding windows within
        #     # the key / value tables, this is helpful to know which
        #     # elements we need to cache.
        #     to_cache, start_idx = [], 0
        #     for prompt_len in self.prompt_lens:
        #         to_cache.extend(
        #             range(
        #                 start_idx + max(0, prompt_len - sliding_window),
        #                 start_idx + prompt_len,
        #             ))
        #         start_idx += self.max_prompt_len
        #     to_cache.extend(range(start_idx, slot_mapping.shape[0]))
        #     self.to_cache = torch.tensor(to_cache,
        #                                  dtype=torch.int32,
        #                                  device=self.slot_mapping.device)

        self.num_prompts = len(context_lens)
        self.num_prompt_tokens = self.num_prompts * self.max_seq_len
        # self.num_generation_tokens = context_lens.shape[0]
        # if block_tables.numel() > 0:
        #     self.max_num_blocks_per_seq = block_tables.shape[1]
        # else:
        #     self.max_num_blocks_per_seq = 0
        # assert block_tables.shape[0] == self.num_generation_tokens
        # self.start_pos = start_pos
        self.kv_cache_dtype = kv_cache_dtype

        # Set during the execution of the first attention op.
        self.attn_bias: Optional[AttentionBias] = None

        self.activation_buffer = ActivationBuffer(model, batched_seq_len)
        self.activation_buffer.allocate_activation_buffer()

    # def __repr__(self) -> str:
    #     # Print only useful metadata.
    #     return (
    #         f'InputMetadata('
    #         f'num_prompt_tokens={self.num_prompt_tokens}, '
    #         f'num_prompts={self.num_prompts}, '
    #         f'prompt_lens={self.prompt_lens}, '
    #         f'num_generation_tokens={self.num_generation_tokens}, '
    #         f'context_lens={self.context_lens}, '
    #         f'max_context_len={self.max_context_len}), '
    #         f'max_num_blocks_per_seq={self.max_num_blocks_per_seq}, '
    #         f'block_tables={self.block_tables}, '
    #         f'selected_token_indices={self.selected_token_indices}, '
    #         f'categorized_sample_indices={self.categorized_sample_indices}, '
    #         f'slot_mapping={self.slot_mapping})')
