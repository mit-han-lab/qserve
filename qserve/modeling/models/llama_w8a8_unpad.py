# File authors: Haotian Tang, Shang Yang, Yujun Lin, Song Han
# @article{lin2024qserve,
#   title={QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving},
#   author={Lin*, Yujun and Tang*, Haotian and Yang*, Shang and Zhang, Zhekai and Xiao, Guangxuan and Gan, Chuang and Han, Song},
#   year={2024}
# }

# Inspired by the following papers:
# @article{touvron2023llama,
#   title={Llama 2: Open foundation and fine-tuned chat models},
#   author={Touvron, Hugo and Martin, Louis and Stone, Kevin and Albert, Peter and Almahairi, Amjad and Babaei, Yasmine and Bashlykov, Nikolay and Batra, Soumya and Bhargava, Prajjwal and Bhosale, Shruti and others},
#   journal={arXiv preprint arXiv:2307.09288},
#   year={2023}
# }

# @article{touvron2023llama,
#   title={Llama: Open and efficient foundation language models},
#   author={Touvron, Hugo and Lavril, Thibaut and Izacard, Gautier and Martinet, Xavier and Lachaux, Marie-Anne and Lacroix, Timoth{\'e}e and Rozi{\`e}re, Baptiste and Goyal, Naman and Hambro, Eric and Azhar, Faisal and others},
#   journal={arXiv preprint arXiv:2302.13971},
#   year={2023}
# }

from typing import Dict, List, Optional

import qserve_backend.fused_attention as fused_attention
import torch
from flash_attn.flash_attn_interface import flash_attn_varlen_func
from qserve_backend import fused_kernels
from torch import nn
from transformers import LlamaConfig

import qserve.utils.constants
from qserve.modeling.layers.activation import SiluAndMulQuant
from qserve.modeling.layers.layernorm import RMSNorm, RMSNormGeneral
from qserve.modeling.layers.quantized_linear import W8A8OF16LinearDynamicInputScale
from qserve.modeling.layers.sampler import Sampler
from qserve.sampling_params import SamplingParams
from qserve.utils.input_metadata import InputMetadata
from qserve.utils.quant_config import QServeQuantConfig
from qserve.utils.weight_utils import (
    convert_pyslice_to_tensor,
    hf_model_weights_iterator,
    load_padded_tensor_parallel_vocab,
    load_tensor_parallel_weights,
)

max_seq_len = qserve.utils.constants.max_seq_len


class LlamaMLP(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        hidden_size = args.hidden_size
        intermediate_size = args.intermediate_size
        self.use_int8 = True

        self.gate_up_proj = W8A8OF16LinearDynamicInputScale(
            hidden_size, 2 * intermediate_size, bias=False
        )
        self.down_proj = W8A8OF16LinearDynamicInputScale(
            intermediate_size, hidden_size, bias=False
        )

        self.act_fn = SiluAndMulQuant(act_sum=False)

    def forward(self, input_metadata: InputMetadata):
        activation_buffer = input_metadata.activation_buffer
        # INT8 in, FP16 out
        self.gate_up_proj(
            activation_buffer.quantized_hidden_states_buffer,
            activation_buffer.quantized_scale_buffer,
            activation_buffer.gate_up_proj_act_buffer,
        )

        # FP16 in, INT8 out
        self.act_fn(
            activation_buffer.gate_up_proj_act_buffer,
            activation_buffer.quantized_mlp_act_buffer,
            activation_buffer.quantized_scale_buffer,
        )

        self.down_proj(
            activation_buffer.quantized_mlp_act_buffer,
            activation_buffer.quantized_scale_buffer,
            activation_buffer.out_down_proj_act_buffer,
        )

    # def forward(self, x, input_scale):
    #     # INT8 in, FP16 out
    #     gate_up, _ = self.gate_up_proj(x, input_scale)
    #     scale = None
    #     x, scale = self.act_fn(gate_up)
    #     x, _ = self.down_proj(x, scale)
    #     return x  # , scale


class LlamaAttention(nn.Module):
    def __init__(
        self, args, layer_idx: int, kv_cache_config: Optional[Dict] = None
    ) -> None:
        super().__init__()
        hidden_size = args.hidden_size
        num_heads = args.num_attention_heads
        num_kv_heads = args.num_key_value_heads
        rope_theta = getattr(args, "rope_theta", 10000)
        rope_scaling = getattr(args, "rope_scaling", None)
        max_position_embeddings = args.max_position_embeddings

        self.layer_idx = layer_idx

        self.hidden_size = hidden_size
        tp_size = 1
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        num_kv_heads_replicas = max(1, tp_size // self.total_num_kv_heads)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.use_int8 = True

        if kv_cache_config is None:
            self.kv_cache_config = {"INT4_ENABLED": False, "ZEROS_ENABLED": False}
            print("[Warning] kv cache config is not provided, using default config KV8")
        else:
            self.kv_cache_config = kv_cache_config

        self.qkv_proj = W8A8OF16LinearDynamicInputScale(
            hidden_size,
            (self.total_num_heads + 2 * self.total_num_kv_heads * num_kv_heads_replicas)
            * self.head_dim,
            bias=False,
        )

        self.o_proj = W8A8OF16LinearDynamicInputScale(
            self.total_num_heads * self.head_dim, hidden_size, bias=False
        )

        self.kv_max_seq_len = min(max_seq_len, self.max_position_embeddings)

        self.invoke_quant = self.invoke_quant_wo_act_sum

    def invoke_quant_wo_act_sum(self, activation_buffer, attn_output):
        fused_kernels.invoke_quant(
            activation_buffer.quantized_hidden_states_buffer,
            attn_output,
            activation_buffer.quantized_scale_buffer,
        )

    def forward(
        self,
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        activation_buffer = input_metadata.activation_buffer
        # INT8 in, FP16 out for this module
        self.qkv_proj(
            activation_buffer.quantized_hidden_states_buffer,
            activation_buffer.quantized_scale_buffer,
            activation_buffer.qkv_proj_act_buffer,
        )
        # qkv = qkv.half()
        if input_metadata.is_prompt:
            # Note: the conversion of kv_scale_orig_quant is currently important
            # by default, self.kv_scale_orig_quant will have the same dtype as the model.
            # but the kernel requires float.
            fused_attention.apply_bias_rope_update_kv_cache(
                activation_buffer.qkv_proj_act_buffer,
                input_metadata.context_lens,
                input_metadata.padding_offsets,  # size [batch_size, max_seq_len]
                input_metadata.block_tables[self.layer_idx],
                self.num_heads,
                self.num_kv_heads,
                input_metadata.max_seq_len,
                64,  # tokens_per_block
                self.hidden_size
                * (1 if self.use_int8 else 2)
                // (2 if self.kv_cache_config["INT4_ENABLED"] else 1),
                self.head_dim,
                10000,
                self.max_position_embeddings,
                True,  # neox style
                self.kv_cache_config["INT4_ENABLED"],  # int4_kv
                self.kv_cache_config["ZEROS_ENABLED"],  # kv_cache_with_zeros
            )

            # FIXME: currently qkv share same scale, plan to use seperate scales
            q, k, v = activation_buffer.qkv_proj_act_buffer.split(
                [self.q_size, self.kv_size, self.kv_size], dim=-1
            )
            # k_cache, v_cache = kv_cache
            q = q.reshape(q.size(0), self.total_num_heads, self.head_dim)
            k = k.reshape(k.size(0), self.num_kv_heads, self.head_dim)
            v = v.reshape(v.size(0), self.num_kv_heads, self.head_dim)

            attn_output = flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q=input_metadata.cu_seqlens,
                cu_seqlens_k=input_metadata.cu_seqlens,
                max_seqlen_q=input_metadata.max_seq_len,
                max_seqlen_k=input_metadata.max_seq_len,
                dropout_p=0.0,
                causal=True,
            )
            attn_output = attn_output.reshape(q.size(0), -1)
        else:
            q, k, v = activation_buffer.qkv_proj_act_buffer.split(
                [self.q_size, self.kv_size, self.kv_size], dim=-1
            )
            q = q.reshape(q.size(0), self.total_num_heads, self.head_dim)
            k = k.reshape(k.size(0), self.num_kv_heads, self.head_dim)
            v = v.reshape(v.size(0), self.num_kv_heads, self.head_dim)
            kv_pointers = input_metadata.block_tables[self.layer_idx]
            # Shang's important fix, but it might cause problem in the end of a block...
            lengths_per_sample = input_metadata.context_lens  # + 1
            alibi_slopes = None
            memory_max_len = self.kv_max_seq_len
            tokens_per_block = 64
            size_per_token = self.hidden_size * (
                1 if self.use_int8 else 2
            )  # size per token
            timestep = input_metadata.max_seq_len
            rotary_embedding_dim = self.head_dim
            rotary_base = 10000
            neox_rotary_style = True
            # shape is (#tokens, #heads, head_dim)
            attn_output = fused_attention.single_query_attention(
                q,
                k,
                v,
                kv_pointers,
                lengths_per_sample,
                alibi_slopes,
                memory_max_len,
                tokens_per_block,
                size_per_token // (2 if self.kv_cache_config["INT4_ENABLED"] else 1),
                timestep,
                rotary_embedding_dim,
                rotary_base,
                neox_rotary_style,
                self.kv_cache_config["INT4_ENABLED"],  # int4_kv
                self.kv_cache_config["ZEROS_ENABLED"],  # kv_cache_with_zeros
            )
            attn_output = attn_output.reshape(q.size(0), -1)
        
        # FP16 in, INT8 out
        self.invoke_quant(activation_buffer, attn_output)
        # INT8 in, FP16 out
        self.o_proj(
            activation_buffer.quantized_hidden_states_buffer,
            activation_buffer.quantized_scale_buffer,
            activation_buffer.out_down_proj_act_buffer,
        )


class LlamaDecoderLayer(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        layer_idx: int,
        kv_cache_config: Optional[Dict] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.use_int8 = True
        # Requires transformers > 4.32.0
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        # self.kv_quant_params = kv_quant_params
        self.self_attn = LlamaAttention(
            config, layer_idx=layer_idx, kv_cache_config=kv_cache_config
        )
        self.mlp = LlamaMLP(config)

        self.input_layernorm = RMSNormGeneral(
            config.hidden_size, act_sum=False, eps=config.rms_norm_eps, use_per_token_quant=True
        )
        self.post_attention_layernorm = RMSNormGeneral(
            config.hidden_size, act_sum=False, eps=config.rms_norm_eps, use_per_token_quant=True
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        # FP16 in FP16 out
        activation_buffer = input_metadata.activation_buffer
        # Self Attention
        residual = hidden_states
        # INT8 quantization
        self.input_layernorm(
            hidden_states,
            activation_buffer.quantized_hidden_states_buffer,
            activation_buffer.quantized_scale_buffer,
        )
        # INT8 -> FP16
        hidden_states = self.self_attn(input_metadata)
        hidden_states = residual + activation_buffer.out_down_proj_act_buffer
        # Fully Connected
        residual = hidden_states
        # FP16 -> INT8
        self.post_attention_layernorm(
            hidden_states,
            activation_buffer.quantized_hidden_states_buffer,
            activation_buffer.quantized_scale_buffer,
        ) # INT8 -> FP16
        self.mlp(input_metadata)
        hidden_states = residual + activation_buffer.out_down_proj_act_buffer
        return hidden_states


class LlamaModel(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        quant_kv_cache: bool = True,
        kv_cache_config: Optional[Dict] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(
            vocab_size,
            config.hidden_size,
        )
        self.layers = nn.ModuleList(
            [
                (
                    LlamaDecoderLayer(config, i, kv_cache_config)
                    if quant_kv_cache
                    else None
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        with torch.no_grad():
            hidden_states = self.embed_tokens(input_ids)
            for i in range(len(self.layers)):
                layer = self.layers[i]
                hidden_states = layer(
                    hidden_states,
                    input_metadata,
                )
            hidden_states = self.norm(hidden_states)

        return hidden_states


class LlamaForCausalLM(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        sampling_params: SamplingParams,
        quant_config: Optional[QServeQuantConfig] = QServeQuantConfig(weight_bits=8),
        kv_cache_config: Optional[Dict] = None,
        quant_path: Optional[str] = None,
    ) -> None:
        quant_kv_cache = True
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.model = LlamaModel(config, quant_kv_cache, kv_cache_config=kv_cache_config)
        vocab_size = config.vocab_size
        # NOTE: The LM head is not quantized.
        self.lm_head = nn.Linear(config.hidden_size, vocab_size, bias=False)
        self._column_parallel_layers = []
        self._row_parallel_layers = ["o_proj", "down_proj"]
        self.sampler = Sampler(sampling_params)

        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads

        self.hidden_size = hidden_size
        tp_size = 1
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        if quant_path is not None:
            self.load_weights(quant_path)

    def forward(
        self,
        input_ids: torch.Tensor,
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, input_metadata)
        output = self.lm_head(hidden_states)  # only compute last logits
        return output.float()

    def sample(
        self,
        input_ids: torch.Tensor,
        logits: torch.Tensor,
        input_metadata: InputMetadata,
    ):
        return self.sampler(input_ids, logits, input_metadata)

    def load_weights(
        self,
        model_name_or_path: str,
        cache_dir: Optional[str] = None,
        load_format: str = "auto",
        revision: Optional[str] = None,
    ):
        if self.quant_config is None:
            col_weight_suffixes = ["weight"]
            row_weight_suffixes = ["weight"]
        else:
            col_weight_suffixes = self.quant_config.get_col_parallel_tensor_names()
            row_weight_suffixes = self.quant_config.get_row_parallel_tensor_names()

        column_parallel_weights: List[str] = []
        for layer in self._column_parallel_layers:
            for suffix in col_weight_suffixes:
                column_parallel_weights.append(f"{layer}.{suffix}")
        row_parallel_weights: List[str] = []
        for layer in self._row_parallel_layers:
            for suffix in row_weight_suffixes:
                row_parallel_weights.append(f"{layer}.{suffix}")

        # TODO fix the tp parallelism
        # tp_size = get_tensor_model_parallel_world_size()
        # tp_rank = get_tensor_model_parallel_rank()
        tp_size = 1
        tp_rank = 0

        q_proj_shard_size = self.config.hidden_size // tp_size
        num_kv_heads_replicas = max(1, tp_size // self.config.num_key_value_heads)
        num_kv_heads_per_gpu = max(1, self.config.num_key_value_heads // tp_size)
        kv_proj_shard_size = (
            self.config.hidden_size
            // self.config.num_attention_heads
            * num_kv_heads_per_gpu
        )
        attention_weight_specs = [
            # (weight_name, shard_size, offset)
            ("q_proj", q_proj_shard_size, 0),
            ("k_proj", kv_proj_shard_size, q_proj_shard_size),
            ("v_proj", kv_proj_shard_size, q_proj_shard_size + kv_proj_shard_size),
        ]
        state_dict = self.state_dict()

        for name, loaded_weight in hf_model_weights_iterator(
            model_name_or_path, cache_dir, load_format, revision
        ):
            if "rotary_emb.inv_freq" in name:
                continue
            # bias is useless for llama
            if "bias" in name:
                continue

            packed_dim = None
            is_transposed = False
            if self.quant_config is not None:
                packed_dim = self.quant_config.get_packed_dim(name)
                is_transposed = self.quant_config.is_transposed(name)
            if is_transposed:
                loaded_weight = convert_pyslice_to_tensor(loaded_weight)
                loaded_weight = loaded_weight.T

            is_attention_weight = False
            for weight_name, shard_size, offset in attention_weight_specs:
                if weight_name not in name:
                    continue
                # print(weight_name)
                param = state_dict[name.replace(weight_name, "qkv_proj")]
                if is_transposed:
                    param = param.T

                if packed_dim is not None:
                    shard_dim = 0 if not is_transposed else 1
                    if packed_dim == shard_dim:
                        shard_size //= self.quant_config.pack_factor
                        offset //= self.quant_config.pack_factor

                if weight_name in ["k_proj", "v_proj"]:
                    shard_id = tp_rank // num_kv_heads_replicas
                else:
                    shard_id = tp_rank
                loaded_weight = loaded_weight[
                    shard_size * shard_id : shard_size * (shard_id + 1)
                ]
                param_slice = param.data[offset : offset + shard_size]
                assert param_slice.shape == loaded_weight.shape

                param_slice.copy_(loaded_weight)
                is_attention_weight = True
                break
            if is_attention_weight:
                continue

            is_gate_up_weight = False
            for stride_id, weight_name in enumerate(["gate_proj", "up_proj"]):
                if weight_name not in name:
                    continue
                param = state_dict[name.replace(weight_name, "gate_up_proj")]
                if is_transposed:
                    param = param.T

                shard_size = param.shape[0] // 2
                loaded_weight = loaded_weight[
                    shard_size * tp_rank : shard_size * (tp_rank + 1)
                ]
                param_slice = param.data[
                    shard_size * stride_id : shard_size * (stride_id + 1)
                ]
                assert param_slice.shape == loaded_weight.shape
                param_slice.copy_(loaded_weight)
                is_gate_up_weight = True
                break
            if is_gate_up_weight:
                continue

            param = state_dict[name]
            if is_transposed:
                param = param.T

            if "embed_tokens" in name or "lm_head" in name:
                load_padded_tensor_parallel_vocab(param, loaded_weight, tp_rank)
                continue

            load_tensor_parallel_weights(
                param,
                loaded_weight,
                name,
                column_parallel_weights,
                row_parallel_weights,
                tp_rank,
            )
