// Inspired by TRT-LLM.
// Modified by Haotian Tang and Shang Yang.
// @article{lin2024qserve,
//   title={QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving},
//   author={Lin*, Yujun and Tang*, Haotian and Yang*, Shang and Zhang, Zhekai and Xiao, Guangxuan and Gan, Chuang and Han, Song},
//   year={2024}
// }
#include <torch/extension.h>
#include "ATen/cuda/CUDAContext.h"
#include <c10/cuda/CUDAGuard.h>

#include "applyBiasRopeUpdateKVCache.h"

INSTANTIATE_ADDFUSEDQKVBIAS_TRANSPOSE(half, KVBlockArray, true);
INSTANTIATE_ADDFUSEDQKVBIAS_TRANSPOSE(half, KVBlockArray, false);
INSTANTIATE_ADDFUSEDQKVBIAS_TRANSPOSE(half, KVLinearBuffer, true);
INSTANTIATE_ADDFUSEDQKVBIAS_TRANSPOSE(half, KVLinearBuffer, false);

void apply_bias_rope_update_kv_cache(const torch::Tensor qkv,
                                     torch::Tensor seq_lens,
                                     torch::Tensor padding_offset,
                                     c10::optional<torch::Tensor> kv_pointers, // B x 2 x M
                                     // virtual sequence length (after padding)
                                     const int head_num,
                                     const int kv_head_num,
                                     const int seq_len,          // max seq len
                                     const int tokens_per_block, // default=64
                                     const int size_per_token,   // default = hidden_size * sizeof(dtype)
                                     const int rotary_embedding_dim,
                                     const float rotary_embedding_base,
                                     const int rotary_embedding_max_positions,
                                     // neox_rotary_style = not interleaved
                                     const bool neox_rotary_style,
                                     const bool int4_kv_cache,
                                     const bool kv_cache_with_zeros)
{
    half *q_ptr = nullptr;
    half *qkv_ptr = reinterpret_cast<half *>(qkv.data_ptr<at::Half>());
    int *seq_lens_ptr = seq_lens.data_ptr<int>();
    int *kv_seq_lens_ptr = seq_lens_ptr;
    // TBD
    int *padding_offset_ptr = padding_offset.data_ptr<int>();
    half *qkv_bias_ptr = nullptr;
    int batch_size = seq_lens.size(0);
    int max_blocks_per_seq = kv_pointers.has_value() ? kv_pointers.value().size(-1) : 0;
    KVBlockArray kvTable(batch_size, max_blocks_per_seq, tokens_per_block, size_per_token);
    kvTable.data = kv_pointers.has_value() ? kv_pointers.value().data_ptr<int64_t>() : nullptr;
    // NOTE: cyclic_kv_cache_len should not be 0.
    int cyclic_kv_cache_len = rotary_embedding_max_positions;
    int sink_token_len = 0;
    int token_num = qkv.size(0);
    // fix this
    int size_per_head = rotary_embedding_dim;
    RotaryScalingType rotary_scale_type = RotaryScalingType::kLINEAR;
    float rotary_embedding_scale = 1.0f;
    PositionEmbeddingType position_embedding_type = PositionEmbeddingType::kROPE_GPT_NEOX;
    int *medusa_position_offsets_ptr = nullptr;
    bool position_shift_enabled = false;
    float *scale_ptr = nullptr;
    int int8_mode = 1;
    KvCacheDataType cache_type;
    if (int4_kv_cache) {
        if (kv_cache_with_zeros)
        {
            cache_type = KvCacheDataType::ZINT4;
        }
        else
        {
            cache_type = KvCacheDataType::INT4;
        }
    }
    else {
        if (kv_cache_with_zeros)
        {
            cache_type = KvCacheDataType::ZINT8;
        }
        else
        {
            cache_type = KvCacheDataType::INT8;
        }
    }
    int beam_width = 1;
    bool enable_paged_kv_fmha = true;
    // TODO: grid_block_cache for different devices??
    int2 grid_block_cache = make_int2(96, 1024);
    auto stream = at::cuda::getCurrentCUDAStream();
    invokeApplyBiasRopeUpdateKVCache<half, KVBlockArray, false>(
        qkv_ptr, q_ptr, kvTable, qkv_bias_ptr, seq_lens_ptr, kv_seq_lens_ptr,
        padding_offset_ptr, batch_size, seq_len, cyclic_kv_cache_len,
        sink_token_len, token_num, head_num, kv_head_num,
        size_per_head, rotary_embedding_dim, rotary_embedding_base,
        rotary_scale_type, rotary_embedding_scale, rotary_embedding_max_positions,
        position_embedding_type, medusa_position_offsets_ptr, position_shift_enabled,
        scale_ptr, int8_mode, cache_type,
        enable_paged_kv_fmha, beam_width, grid_block_cache, stream);

    // void invokeApplyBiasRopeUpdateKVCache<T, KVCacheBuffer, IS_GENERATE>(T * QKV, T * Q,
    //     KVCacheBuffer & kvTable, const T* qkv_bias, const int* seq_lens, const int* kv_seq_lens,
    //     const int* padding_offset, const int batch_size, const int seq_len, const int cyclic_kv_cache_len,
    //     const int sink_token_len, const int token_num, const int head_num, const int kv_head_num,
    //     const int size_per_head, const int rotary_embedding_dim, const float rotary_embedding_base,
    //     const RotaryScalingType rotary_scale_type, const float rotary_embedding_scale,
    //     const int rotary_embedding_max_positions, const PositionEmbeddingType position_embedding_type,
    //     const int* medusa_position_offsets, const bool position_shift_enabled, const float* scale,
    //     const int int8_mode, const KvCacheDataType cache_type, const float* kvScaleOrigQuant,
    //     const bool enable_paged_kv_fmha, const int beam_width, int2& grid_block_cache, cudaStream_t stream);
}