// Inspired by TRT-LLM.
// Modified by Haotian Tang and Shang Yang.
// @article{lin2024qserve,
//   title={QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving},
//   author={Lin*, Yujun and Tang*, Haotian and Yang*, Shang and Zhang, Zhekai and Xiao, Guangxuan and Gan, Chuang and Han, Song},
//   journal={arXiv preprint arXiv:2405.04532},
//   year={2024}
// }
#pragma once
#include "cudaTypeUtils.cuh"
#include "decoderMaskedMultiheadAttentionUtils.h"
#include "gptKernels.h"
#include "kvCacheUtils.h"

#define WARP_SIZE 32
#define HALF_WARP_SIZE 16

template <typename T, int Dh_MAX>
struct Rotary_vec_t
{
    using Type = T;
    using Packed_type = T;
    static constexpr int size = 1;
};

template <>
struct Rotary_vec_t<float, 32>
{
    using Type = float;
    using Packed_type = float;
    static constexpr int size = 1;
};

template <>
struct Rotary_vec_t<float, 64>
{
    using Type = float2;
    using Packed_type = float2;
    static constexpr int size = 2;
};

template <>
struct Rotary_vec_t<float, 128>
{
    using Type = float4;
    using Packed_type = float2;
    static constexpr int size = 4;
};

template <>
struct Rotary_vec_t<float, 256>
{
    using Type = mmha::Float8_;
    using Packed_type = float2;
    static constexpr int size = 8;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
struct Rotary_vec_t<half, 32>
{
    using Type = uint16_t;
    using Packed_type = uint16_t;
    static constexpr int size = 1;
};

template <>
struct Rotary_vec_t<half, 64>
{
    using Type = uint32_t;
    using Packed_type = uint32_t;
    static constexpr int size = 2;
};

template <>
struct Rotary_vec_t<half, 128>
{
    using Type = uint2;
    using Packed_type = uint32_t;
    static constexpr int size = 4;
};

template <>
struct Rotary_vec_t<half, 256>
{
    using Type = uint4;
    using Packed_type = uint32_t;
    static constexpr int size = 8;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, typename T_cache, int Dh_MAX, bool ADD_BIAS, bool STORE_QKV, bool POS_SHIFT,
          typename KVCacheBuffer, bool IS_GENERATE, bool INT4KV = false, bool KV_CACHE_WITH_ZEROS = false>
__global__ void applyBiasRopeUpdateKVCache(T *QKV, T *Q, KVCacheBuffer kvCacheBuffer, const T *__restrict qkv_bias,
                                           const int *seq_lens, const int *kv_seq_lens, const int *padding_offset,
                                           const int num_tokens, const int batch_size, const int seq_len, const int cyclic_kv_cache_len,
                                           const int sink_token_len, const int head_num, const int kv_head_num, const int qheads_per_kv_head,
                                           const int size_per_head, const int rotary_embedding_dim, float rotary_embedding_base,
                                           RotaryScalingType const rotary_scale_type, float rotary_embedding_scale, const int rotary_embedding_max_positions,
                                           PositionEmbeddingType const position_embedding_type, const int *medusa_position_offsets, const int beam_width)
{
    // This kernel add bias to QKV, which has shape [batch_size, seq_len, 3, head_num, size_per_head]
    // Extract the Q input when using paged KV FMHA.
    // For q and k, also apply the rotary embedding.

    // NOTE:
    // head_num == kv_head_num
    //   QKV src shape (batch_size, seq_len, 3, head_num, size_per_head)
    //                  ^^^^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^^^^^^^^^
    //                           m                        n
    //   QKV dst shape (3, batch_size, head_num, seq_len, size_per_head)
    // head_num != kv_head_num
    //   QKV src shape: (batch_size, seq_len, head_num * size_per_head + 2 * kv_head_num * size_per_head)
    //                   ^^^^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    //                             m                               n
    //   Q dst shape: (batch_size, head_num, seq_len, size_per_head)
    //   KV dst shape: (batch_size, kv_head_num, seq_len, size_per_head)

    // There are two kinds of output:
    //  1. Contiguous QKV output.
    //  2. Contiguous Q output + Paged KV output (needed by Paged KV FMHA kernels).

    // VEC_SIZE is power of 2.
    constexpr int VEC_SIZE = Rotary_vec_t<T, Dh_MAX>::size;
    using Vec_type = typename Rotary_vec_t<T, Dh_MAX>::Type;
    using Packed_type = typename Rotary_vec_t<T, Dh_MAX>::Packed_type;
    const bool has_padding = padding_offset == nullptr;

    constexpr bool ENABLE_8BITS_CACHE = sizeof(T_cache) == 1;
    constexpr bool ENABLE_4BITS_CACHE = INT4KV && ENABLE_8BITS_CACHE;
    constexpr bool ENABLE_ZEROS = KV_CACHE_WITH_ZEROS;
    const int sizePerHeadDivX = size_per_head / VEC_SIZE;
    using T_dst = T_cache;

    const int head_idx = blockIdx.y;
    // Block size is always 32 in the x dimension.
    int tidx = threadIdx.x;
    // The half head dimension for remapping.
    // 32 threads in one warp
    // (first rotary threads + first no rotary threads) = first 16 threads
    // (second rotary threads + second no rotary threads) = second 16 threads
    const int half_within_bound_dim = size_per_head / 2;
    const int half_rotary_embedding_dim = rotary_embedding_dim / 2;
    const int half_rotary_embedding_threads = rotary_embedding_dim / (2 * VEC_SIZE);
    const int half_non_rotary_embedding_threads = (size_per_head - rotary_embedding_dim) / (2 * VEC_SIZE);
    const int threads_per_head = size_per_head / VEC_SIZE;
    // Remap to the correct half head size when head size is not power of 2.
    // This is mianly designed for the gptneox_style_rotary_embedding (which rotates the half embedding.)
    // The first 16 threads will handle the first half head size.
    const bool first_half = tidx < HALF_WARP_SIZE;
    const int second_half = !first_half;

    int rotary_local_tidx = (tidx - second_half * HALF_WARP_SIZE);

    // Three partitions for each half threads.
    //  apply rotary (tidx * VEC_SIZE < half_rotary_embdding)
    //  don't apply rotary= (half_rotary_embedding <= tidx * VEC_SIZE < half_size_per_head)
    //  out of the bound (tidx * VEC_SIZE >= half_size_per_head)
    tidx = rotary_local_tidx * VEC_SIZE >= half_within_bound_dim
               ? -1
               : (rotary_local_tidx * VEC_SIZE < half_rotary_embedding_dim
                      ? (rotary_local_tidx + second_half * half_rotary_embedding_threads)
                      : (rotary_local_tidx + half_rotary_embedding_threads + second_half * half_non_rotary_embedding_threads));

    const int hidden_size = head_num * size_per_head;
    const int hidden_idx = head_idx * size_per_head + tidx * VEC_SIZE;
    const int kv_head_idx = head_idx / qheads_per_kv_head;
    const int hidden_idx_kv = kv_head_idx * size_per_head + tidx * VEC_SIZE;
    const int n = (head_num + 2 * kv_head_num) * size_per_head;
    const int src_k_offset = hidden_size;
    const int src_v_offset = hidden_size + kv_head_num * size_per_head;

    // Dynamic scaling of rotary embedding.
    const bool dynamic_scale = rotary_scale_type == RotaryScalingType::kDYNAMIC;

    for (int token_idx = blockIdx.x * blockDim.y + threadIdx.y; token_idx < num_tokens;
         token_idx += gridDim.x * blockDim.y)
    {
        // The index of the token in the batch. It includes "virtual" padding (even if the input is not padded)
        // such that the sequence index and the position in the sequence can be obtained using the max.
        // sequence length as:
        const int token_padding_offset = (has_padding || IS_GENERATE) ? 0 : padding_offset[token_idx];
        const int global_token_idx = token_idx + token_padding_offset;
        const int batch_beam_idx = global_token_idx / seq_len;
        // half *k_scale_quant_orig_ptr = k_scale_quant_orig[batch_beam_idx];
        // half *v_scale_quant_orig_ptr = v_scale_quant_orig[batch_beam_idx];
        // TODO: optimize this for generation by using anther dimension of grid.
        const int seq_idx = global_token_idx % seq_len;
        const int final_kv_seq_len = (!IS_GENERATE) ? kv_seq_lens[batch_beam_idx] : 0;
        const int actual_seq_len = seq_lens[batch_beam_idx];
        // Chunked attention: takes past_kv_sequence_length into consideration.
        const int token_idx_in_seq = (!IS_GENERATE) ? (final_kv_seq_len - actual_seq_len) + seq_idx : (actual_seq_len - seq_len + seq_idx);
        const bool valid_seq = IS_GENERATE || (token_idx_in_seq < actual_seq_len || !has_padding);
        // NOTE: only Medusa needs the position offsets.
        // In the generation phase, we assume all sequences should have the same input length.
        const int rotary_position = token_idx_in_seq;

        // only update the base and/or scale if needed based on scale_type
        // we have already updated the scale in host if it is linear scale.
        float2 updated_base_scale = mmha::update_dynamic_scaling_rotary(rotary_embedding_base, rotary_embedding_scale,
                                                                        actual_seq_len, rotary_embedding_max_positions, rotary_embedding_dim, dynamic_scale);
        const float updated_base = updated_base_scale.x;
        const float updated_scale = updated_base_scale.y;

        const bool is_masked = !valid_seq || tidx < 0;

        // head_num == kv_head_num:
        //   src QKV: [batch, time, 3, head_num, size_per_head]
        // head_num != kv_head_num:
        //   src QKV: [batch, time, head_num * size_per_head + 2 * kv_head_num * size_per_head]
        auto const src_q_idx = static_cast<size_t>(token_idx) * n + hidden_idx;
        auto const src_k_idx = static_cast<size_t>(token_idx) * n + src_k_offset + hidden_idx_kv;
        auto const src_v_idx = static_cast<size_t>(token_idx) * n + src_v_offset + hidden_idx_kv;

        Vec_type q, k, v;
        Vec_type q_bias, k_bias, v_bias;
        // key without position embedding
        Vec_type k_wo_pos;

        // load q,k,v and add bias
        if (!is_masked)
        {
            q = *reinterpret_cast<const Vec_type *>(&QKV[src_q_idx]);
            k = *reinterpret_cast<const Vec_type *>(&QKV[src_k_idx]);
            v = *reinterpret_cast<const Vec_type *>(&QKV[src_v_idx]);

            if constexpr (ADD_BIAS)
            {
                q_bias = *reinterpret_cast<const Vec_type *>(&qkv_bias[hidden_idx]);
                k_bias = *reinterpret_cast<const Vec_type *>(&qkv_bias[hidden_idx_kv + src_k_offset]);
                v_bias = *reinterpret_cast<const Vec_type *>(&qkv_bias[hidden_idx_kv + src_v_offset]);

                q = mmha::add(q, q_bias);
                k = mmha::add(k, k_bias);
                v = mmha::add(v, v_bias);
            }
            k_wo_pos = k;
        }

        // Rotary Emedding.
        switch (position_embedding_type)
        {
        // Rotate every two elements (need at two elements per thead).
        // e.g.  0  1  2  3  4  5  6  7 (head size 8)
        //      -1  0 -3  2 -5  4 -7  6
        case PositionEmbeddingType::kROPE_GPTJ:
        {
            mmha::apply_rotary_embedding(
                q, k, tidx, rotary_embedding_dim, updated_base, updated_scale, rotary_position);
            break;
        }
        // Rotate by half rotary embedding.
        // e.g.  0  1  2  3  4  5  6  7 (head size 8)
        //      -4 -5 -6 -7  0  1  2  3
        case PositionEmbeddingType::kROPE_GPT_NEOX:
        {
            // One warp of threads handle one head.
            // where the first 16 threads process the first half of rotary embedding,
            //  and second 16 threads process the second half.
            // Note that the half rotary embedding may not be power of 2.
            // e.g. 80 head size (next power of 2 is 128, so each thread will process 4 elements),
            //  which means only thread 0 ~ 10 (exclusive), and 16 ~ 26 (exclusive) have work to do.
            mmha::apply_rotary_embedding_gptneox<Vec_type, Packed_type, T>(
                q, k, tidx, rotary_embedding_dim, updated_base, updated_scale, rotary_position, first_half);
            break;
        }
        }

        const int channelIdx{tidx};
        const int tokenIdxLowerBound = max(actual_seq_len - cyclic_kv_cache_len + sink_token_len, sink_token_len);
        const bool valid_kv_cache_pos = kvCacheBuffer.data != nullptr // In KV-cache-less mode. No need to store KV values
                                        && (token_idx_in_seq >= tokenIdxLowerBound || token_idx_in_seq < sink_token_len);
        const int token_kv_idx = token_idx_in_seq; // kvCacheBuffer.getKVTokenIdx(token_idx_in_seq);

        auto kDst = reinterpret_cast<T_dst *>(kvCacheBuffer.getKBlockPtr(batch_beam_idx, token_kv_idx));
        auto vDst = reinterpret_cast<T_dst *>(kvCacheBuffer.getVBlockPtr(batch_beam_idx, token_kv_idx));
        half *k_scale_block_ptr = reinterpret_cast<half *>(kDst + kvCacheBuffer.mBytesPerSeq);
        half *v_scale_block_ptr = reinterpret_cast<half *>(vDst + kvCacheBuffer.mBytesPerSeq);
        // half *k_scale_cur_ptr = k_scale_block_ptr + kvCacheBuffer.getLocalIdx(token_kv_idx) * kv_head_num + kv_head_idx;
        // half *v_scale_cur_ptr = v_scale_block_ptr + kvCacheBuffer.getLocalIdx(token_kv_idx) * kv_head_num + kv_head_idx;
        half *k_scale_cur_ptr = k_scale_block_ptr + kv_head_idx * kvCacheBuffer.mTokensPerBlock + kvCacheBuffer.getLocalIdx(token_kv_idx);
        half *v_scale_cur_ptr = v_scale_block_ptr + kv_head_idx * kvCacheBuffer.mTokensPerBlock + kvCacheBuffer.getLocalIdx(token_kv_idx);
        // on-the-fly quantization and scaling factor calculation
        float k_max, v_max, k_min, v_min, cur_k_scale_orig_quant, cur_v_scale_orig_quant, cur_k_zeros, cur_v_zeros;

        if constexpr (ENABLE_ZEROS)
        {
            half *k_zero_cur_ptr = k_scale_cur_ptr + kv_head_num * kvCacheBuffer.mTokensPerBlock;
            half *v_zero_cur_ptr = v_scale_cur_ptr + kv_head_num * kvCacheBuffer.mTokensPerBlock;
            k_max = mmha::vec_max_no_abs<Vec_type>(k);
            v_max = mmha::vec_max_no_abs<Vec_type>(v);
            k_min = mmha::vec_min_no_abs<Vec_type>(k);
            v_min = mmha::vec_min_no_abs<Vec_type>(v);
            assert(threads_per_head <= WARP_SIZE);
#pragma unroll
            for (int mask = threads_per_head / 2; mask >= 1; mask /= 2)
            {
                k_max = fmaxf(k_max, __shfl_xor_sync(mmha::shfl_mask(threads_per_head), k_max, mask));
                v_max = fmaxf(v_max, __shfl_xor_sync(mmha::shfl_mask(threads_per_head), v_max, mask));
                k_min = fminf(k_min, __shfl_xor_sync(mmha::shfl_mask(threads_per_head), k_min, mask));
                v_min = fminf(v_min, __shfl_xor_sync(mmha::shfl_mask(threads_per_head), v_min, mask));
            }
            // wb to DRAM
            if constexpr (ENABLE_4BITS_CACHE)
            {
                if (tidx == 0)
                {
                    *k_scale_cur_ptr = __float2half_rn((k_max - k_min) / 15);
                    *v_scale_cur_ptr = __float2half_rn((v_max - v_min) / 15);
                    *k_zero_cur_ptr = __float2half_rn(-15.0f * k_min / (k_max - k_min));
                    *v_zero_cur_ptr = __float2half_rn(-15.0f * v_min / (v_max - v_min));
                }
            }
            else
            {
                if (tidx == 0)
                {
                    *k_scale_cur_ptr = __float2half_rn((k_max - k_min) / 255);
                    *v_scale_cur_ptr = __float2half_rn((v_max - v_min) / 255);
                    *k_zero_cur_ptr = __float2half_rn(-255.0f * k_min / (k_max - k_min));
                    *v_zero_cur_ptr = __float2half_rn(-255.0f * v_min / (v_max - v_min));
                }
            }
            __syncthreads();
            cur_k_scale_orig_quant = 1.0f / __half2float(*k_scale_cur_ptr);
            cur_v_scale_orig_quant = 1.0f / __half2float(*v_scale_cur_ptr);
            cur_k_zeros = __half2float(*k_zero_cur_ptr);
            cur_v_zeros = __half2float(*v_zero_cur_ptr);
        }
        else
        {
            k_max = mmha::vec_max<Vec_type>(k);
            v_max = mmha::vec_max<Vec_type>(v);
            // tree reduction for final results (within a warp)
            assert(threads_per_head <= WARP_SIZE);
#pragma unroll
            for (int mask = threads_per_head / 2; mask >= 1; mask /= 2)
            {
                k_max = fmaxf(k_max, __shfl_xor_sync(mmha::shfl_mask(threads_per_head), k_max, mask));
                v_max = fmaxf(v_max, __shfl_xor_sync(mmha::shfl_mask(threads_per_head), v_max, mask));
            }
            // wb to DRAM
            if constexpr (ENABLE_4BITS_CACHE)
            {
                if (tidx == 0)
                {
                    *k_scale_cur_ptr = __float2half_rn(k_max / 7);
                    *v_scale_cur_ptr = __float2half_rn(v_max / 7);
                }
            }
            else
            {
                if (tidx == 0)
                {
                    *k_scale_cur_ptr = __float2half_rn(k_max / 127);
                    *v_scale_cur_ptr = __float2half_rn(v_max / 127);
                }
            }
            __syncthreads();
            cur_k_scale_orig_quant = 1.0f / __half2float(*k_scale_cur_ptr);
            cur_v_scale_orig_quant = 1.0f / __half2float(*v_scale_cur_ptr);
        }

        if (!is_masked)
        {
            int inBlockIdx;

            if constexpr (ENABLE_8BITS_CACHE)
            {
                inBlockIdx = kvCacheBuffer.getKVLocalIdx(token_kv_idx, kv_head_idx, sizePerHeadDivX, channelIdx);
            }
            Vec_type k_to_cache = (POS_SHIFT) ? k_wo_pos : k;

            if constexpr (STORE_QKV)
            {
                *reinterpret_cast<Vec_type *>(&QKV[src_q_idx]) = q;
            }
            else
            {
                *reinterpret_cast<Vec_type *>(&Q[token_idx * head_num * size_per_head + hidden_idx]) = q;
            }
            if ((head_num == kv_head_num) || (head_idx == (kv_head_idx * qheads_per_kv_head)))
            {
                if constexpr (STORE_QKV)
                {
                    *reinterpret_cast<Vec_type *>(&QKV[src_k_idx]) = k;
                    if constexpr (ADD_BIAS)
                    {
                        *reinterpret_cast<Vec_type *>(&QKV[src_v_idx]) = v;
                    }
                }

                if (valid_kv_cache_pos)
                {
                    if (ENABLE_ZEROS)
                    {
                        if constexpr (ENABLE_4BITS_CACHE)
                        {
                            inBlockIdx = inBlockIdx * VEC_SIZE / 2;
                            // Store 8bits kv cache.
                            mmha::store_4bits_kv_cache_vec(kDst, k_to_cache, inBlockIdx, cur_k_scale_orig_quant, cur_k_zeros);
                            mmha::store_4bits_kv_cache_vec(vDst, v, inBlockIdx, cur_v_scale_orig_quant, cur_v_zeros);
                        }
                        else if constexpr (ENABLE_8BITS_CACHE)
                        {
                            inBlockIdx = inBlockIdx * VEC_SIZE;
                            // Cast float scale to dst data type.
                            // using T_scale = typename mmha::kv_cache_scale_type_t<T, T_cache>::Type;
                            // T_scale kscaleOrigQuant, vscaleOrigQuant;
                            // mmha::convert_from_float(&kscaleOrigQuant, kvScaleOrigQuant[0]);
                            // mmha::convert_from_float(&vscaleOrigQuant, kvScaleOrigQuant[1]);
                            // Store 8bits kv cache.
                            mmha::store_8bits_kv_cache_vec(kDst, k_to_cache, inBlockIdx, cur_k_scale_orig_quant, cur_k_zeros);
                            mmha::store_8bits_kv_cache_vec(vDst, v, inBlockIdx, cur_v_scale_orig_quant, cur_v_zeros);
                        }
                        else
                        {
                            reinterpret_cast<Vec_type *>(kDst)[inBlockIdx] = k_to_cache;
                            reinterpret_cast<Vec_type *>(vDst)[inBlockIdx] = v;
                        }
                    }
                    else
                    {
                        if constexpr (ENABLE_4BITS_CACHE)
                        {
                            inBlockIdx = inBlockIdx * VEC_SIZE / 2;
                            // Store 8bits kv cache.
                            mmha::store_4bits_kv_cache_vec(kDst, k_to_cache, inBlockIdx, cur_k_scale_orig_quant);
                            mmha::store_4bits_kv_cache_vec(vDst, v, inBlockIdx, cur_v_scale_orig_quant);
                        }
                        else if constexpr (ENABLE_8BITS_CACHE)
                        {
                            inBlockIdx = inBlockIdx * VEC_SIZE;
                            // Cast float scale to dst data type.
                            // using T_scale = typename mmha::kv_cache_scale_type_t<T, T_cache>::Type;
                            // T_scale kscaleOrigQuant, vscaleOrigQuant;
                            // mmha::convert_from_float(&kscaleOrigQuant, kvScaleOrigQuant[0]);
                            // mmha::convert_from_float(&vscaleOrigQuant, kvScaleOrigQuant[1]);
                            // Store 8bits kv cache.
                            mmha::store_8bits_kv_cache_vec(kDst, k_to_cache, inBlockIdx, cur_k_scale_orig_quant);
                            mmha::store_8bits_kv_cache_vec(vDst, v, inBlockIdx, cur_v_scale_orig_quant);
                        }
                        else
                        {
                            reinterpret_cast<Vec_type *>(kDst)[inBlockIdx] = k_to_cache;
                            reinterpret_cast<Vec_type *>(vDst)[inBlockIdx] = v;
                        }
                    }
                }
            }
        }
    }
}

// Grid_block_cache (grid dim, block dim).
// This caches the block_size, grid_size calculated by cudaOccupancyMaxPotentialBlockSize.
#define APPLY_BIAS_ROPE_UPDATE_KV_CACHE(Dh_MAX, ADD_BIAS, STORE_QKV, POS_SHIFT, INT4KV, KV_CACHE_WITH_ZEROS)                                \
    int block_size = grid_block_cache.x, grid_size = grid_block_cache.y;                                                                    \
    int tokens_per_block = (block_size + WARP_SIZE - 1) / WARP_SIZE;                                                                        \
    dim3 block(WARP_SIZE, tokens_per_block);                                                                                                \
    int blocks_per_sequence = std::min((grid_size + head_num - 1) / head_num, (token_num + tokens_per_block - 1) / tokens_per_block);       \
    dim3 grid(blocks_per_sequence, head_num);                                                                                               \
    applyBiasRopeUpdateKVCache<T, T_cache, Dh_MAX, ADD_BIAS, STORE_QKV, POS_SHIFT, KVCacheBuffer, IS_GENERATE, INT4KV, KV_CACHE_WITH_ZEROS> \
        <<<grid, block, 0, stream>>>(QKV, Q, kvTable, qkv_bias, seq_lens, kv_seq_lens, padding_offset,                                      \
                                     token_num, batch_size, seq_len, cyclic_kv_cache_len, sink_token_len, head_num,                         \
                                     kv_head_num, head_num / kv_head_num, size_per_head, rotary_embedding_dim, rotary_embedding_base,       \
                                     rotary_scale_type, updated_rotary_embedding_scale, rotary_embedding_max_positions,                     \
                                     position_embedding_type, medusa_position_offsets, beam_width);

template <int Dh_MAX, typename T, typename T_cache, typename KVCacheBuffer, bool IS_GENERATE, bool INT4KV, bool KV_CACHE_WITH_ZEROS>
void kernelDispatchHeadSize(T *QKV, T *Q, KVCacheBuffer &kvTable, const T *qkv_bias, const int *seq_lens,
                            const int *kv_seq_lens, const int *padding_offset, const int batch_size, const int seq_len,
                            const int cyclic_kv_cache_len, const int sink_token_len, const int token_num, const int head_num,
                            const int kv_head_num, const int size_per_head, const int rotary_embedding_dim, const float rotary_embedding_base,
                            const RotaryScalingType rotary_scale_type, const float rotary_embedding_scale,
                            const int rotary_embedding_max_positions, const PositionEmbeddingType position_embedding_type,
                            const int *medusa_position_offsets, const bool position_shift_enabled, const float *scale,
                            const int int8_mode, const bool enable_paged_kv_fmha, const int beam_width,
                            int2 &grid_block_cache, cudaStream_t stream)
{
    const bool add_bias = qkv_bias != nullptr;
    const bool store_contiguous_qkv = true; //! enable_paged_kv_fmha;

    // Update scale if scale_type == RotaryScalingType::kLINEAR.
    const float updated_rotary_embedding_scale = rotary_scale_type == RotaryScalingType::kLINEAR ? 1.0f / rotary_embedding_scale : rotary_embedding_scale;

    if (add_bias)
    {
        // second template param is STORE_QKV, which should always be true.
        if (position_shift_enabled)
        {
            APPLY_BIAS_ROPE_UPDATE_KV_CACHE(Dh_MAX, true, true, true, INT4KV, KV_CACHE_WITH_ZEROS);
        }
        else
        {
            APPLY_BIAS_ROPE_UPDATE_KV_CACHE(Dh_MAX, true, true, false, INT4KV, KV_CACHE_WITH_ZEROS);
        }
    }
    else
    {
        // second template param is STORE_QKV, which should always be true.
        if (position_shift_enabled)
        {
            APPLY_BIAS_ROPE_UPDATE_KV_CACHE(Dh_MAX, false, true, true, INT4KV, KV_CACHE_WITH_ZEROS);
        }
        else
        {
            APPLY_BIAS_ROPE_UPDATE_KV_CACHE(Dh_MAX, false, true, false, INT4KV, KV_CACHE_WITH_ZEROS);
        }
    }
}

template <typename T, typename T_cache, typename KVCacheBuffer, bool IS_GENERATE, bool INT4KV, bool KV_CACHE_WITH_ZEROS>
void invokeApplyBiasRopeUpdateKVCacheDispatch(T *QKV, T *Q, KVCacheBuffer &kvTable, const T *qkv_bias,
                                              const int *seq_lens, const int *kv_seq_lens, const int *padding_offset, const int batch_size, const int seq_len,
                                              const int cyclic_kv_cache_len, const int sink_token_len, const int token_num, const int head_num,
                                              const int kv_head_num, const int size_per_head, const int rotary_embedding_dim, const float rotary_embedding_base,
                                              const RotaryScalingType rotary_scale_type, const float rotary_embedding_scale,
                                              const int rotary_embedding_max_positions, const PositionEmbeddingType position_embedding_type,
                                              const int *medusa_position_offsets, const bool position_shift_enabled, const float *scale,
                                              const int int8_mode, const bool enable_paged_kv_fmha, const int beam_width,
                                              int2 &grid_block_cache, cudaStream_t stream)
{
    if (int8_mode == 2)
    {
        printf("w8a8 not yet implemented with RoPE\n"); // TODO
        return;
    }
    if (IS_GENERATE)
    {
        // NOTE: generation phase may have input sequence length > 1 under the medusa mode.
        if (padding_offset != nullptr)
        {
            printf("Generation phase should not use padding_offset\n");
            return;
        }
    }

    // Use specialized kernels for different heads (better balance of work).
    if (size_per_head % 8 != 0)
    {
        printf("Head size needs to be multiple of 8!\n");
        return;
    }
    if (rotary_embedding_dim % 8 != 0)
    {
        printf("Rotary embedding dimension needs to be multiple of 8!\n");
        return;
    }
    // GPTJ Rotary embedding needs at least two elements per thread.
    if (size_per_head <= 64)
    {
        kernelDispatchHeadSize<64, T, T_cache, KVCacheBuffer, IS_GENERATE, INT4KV, KV_CACHE_WITH_ZEROS>(QKV, Q, kvTable, qkv_bias, seq_lens,
                                                                                                        kv_seq_lens, padding_offset, batch_size, seq_len, cyclic_kv_cache_len, sink_token_len, token_num, head_num,
                                                                                                        kv_head_num, size_per_head, rotary_embedding_dim, rotary_embedding_base, rotary_scale_type,
                                                                                                        rotary_embedding_scale, rotary_embedding_max_positions, position_embedding_type, medusa_position_offsets,
                                                                                                        position_shift_enabled, scale,
                                                                                                        int8_mode, enable_paged_kv_fmha, beam_width,
                                                                                                        grid_block_cache, stream);
    }
    else if (size_per_head <= 128)
    {
        kernelDispatchHeadSize<128, T, T_cache, KVCacheBuffer, IS_GENERATE, INT4KV, KV_CACHE_WITH_ZEROS>(QKV, Q, kvTable, qkv_bias, seq_lens,
                                                                                                         kv_seq_lens, padding_offset, batch_size, seq_len, cyclic_kv_cache_len, sink_token_len, token_num, head_num,
                                                                                                         kv_head_num, size_per_head, rotary_embedding_dim, rotary_embedding_base, rotary_scale_type,
                                                                                                         rotary_embedding_scale, rotary_embedding_max_positions, position_embedding_type, medusa_position_offsets,
                                                                                                         position_shift_enabled, scale,
                                                                                                         int8_mode, enable_paged_kv_fmha, beam_width,
                                                                                                         grid_block_cache, stream);
    }
    else if (size_per_head <= 256)
    {
        kernelDispatchHeadSize<256, T, T_cache, KVCacheBuffer, IS_GENERATE, INT4KV, KV_CACHE_WITH_ZEROS>(QKV, Q, kvTable, qkv_bias, seq_lens,
                                                                                                         kv_seq_lens, padding_offset, batch_size, seq_len, cyclic_kv_cache_len, sink_token_len, token_num, head_num,
                                                                                                         kv_head_num, size_per_head, rotary_embedding_dim, rotary_embedding_base, rotary_scale_type,
                                                                                                         rotary_embedding_scale, rotary_embedding_max_positions, position_embedding_type, medusa_position_offsets,
                                                                                                         position_shift_enabled, scale,
                                                                                                         int8_mode, enable_paged_kv_fmha, beam_width, grid_block_cache, stream);
    }
    else
    {
        printf("applyBiasRopeUpdateKVCache kernel doesn't support head size = %d\n", size_per_head);
        return;
    }
}

template <typename T, typename KVCacheBuffer, bool IS_GENERATE>
void invokeApplyBiasRopeUpdateKVCache(T *QKV, T *Q, KVCacheBuffer &kvTable, const T *qkv_bias, const int *seq_lens,
                                      const int *kv_seq_lens, const int *padding_offset, const int batch_size, const int seq_len,
                                      const int cyclic_kv_cache_len, const int sink_token_len, const int token_num, const int head_num,
                                      const int kv_head_num, const int size_per_head, const int rotary_embedding_dim, const float rotary_embedding_base,
                                      const RotaryScalingType rotary_scale_type, const float rotary_embedding_scale,
                                      const int rotary_embedding_max_positions, const PositionEmbeddingType position_embedding_type,
                                      const int *medusa_position_offsets, const bool position_shift_enabled, const float *scale, const int int8_mode,
                                      const KvCacheDataType cache_type,
                                      const bool enable_paged_kv_fmha,
                                      const int beam_width, int2 &grid_block_cache, cudaStream_t stream)
{
    // Block handles both K and V tile.
    constexpr int x = (sizeof(T) == 4) ? 4 : 8;
    if (size_per_head % x != 0)
    {
        printf("Size per head is not a multiple of X\n");
        return;
    }

    if (cache_type == KvCacheDataType::INT4)
    {
        invokeApplyBiasRopeUpdateKVCacheDispatch<T, int8_t, KVCacheBuffer, IS_GENERATE, true, false>(QKV, Q, kvTable, qkv_bias,
                                                                                                     seq_lens, kv_seq_lens, padding_offset, batch_size, seq_len, cyclic_kv_cache_len, sink_token_len, token_num,
                                                                                                     head_num, kv_head_num, size_per_head, rotary_embedding_dim, rotary_embedding_base, rotary_scale_type,
                                                                                                     rotary_embedding_scale, rotary_embedding_max_positions, position_embedding_type, medusa_position_offsets,
                                                                                                     position_shift_enabled, scale,
                                                                                                     int8_mode, enable_paged_kv_fmha, beam_width,
                                                                                                     grid_block_cache, stream);
    }

    else if (cache_type == KvCacheDataType::ZINT4)
    {
        invokeApplyBiasRopeUpdateKVCacheDispatch<T, int8_t, KVCacheBuffer, IS_GENERATE, true, true>(QKV, Q, kvTable, qkv_bias,
                                                                                                    seq_lens, kv_seq_lens, padding_offset, batch_size, seq_len, cyclic_kv_cache_len, sink_token_len, token_num,
                                                                                                    head_num, kv_head_num, size_per_head, rotary_embedding_dim, rotary_embedding_base, rotary_scale_type,
                                                                                                    rotary_embedding_scale, rotary_embedding_max_positions, position_embedding_type, medusa_position_offsets,
                                                                                                    position_shift_enabled, scale,
                                                                                                    int8_mode, enable_paged_kv_fmha, beam_width,
                                                                                                    grid_block_cache, stream);
    }

    else if (cache_type == KvCacheDataType::INT8)
    {
        invokeApplyBiasRopeUpdateKVCacheDispatch<T, int8_t, KVCacheBuffer, IS_GENERATE, false, false>(QKV, Q, kvTable, qkv_bias,
                                                                                                      seq_lens, kv_seq_lens, padding_offset, batch_size, seq_len, cyclic_kv_cache_len, sink_token_len, token_num,
                                                                                                      head_num, kv_head_num, size_per_head, rotary_embedding_dim, rotary_embedding_base, rotary_scale_type,
                                                                                                      rotary_embedding_scale, rotary_embedding_max_positions, position_embedding_type, medusa_position_offsets,
                                                                                                      position_shift_enabled, scale,
                                                                                                      int8_mode, enable_paged_kv_fmha, beam_width,
                                                                                                      grid_block_cache, stream);
    }

    else if (cache_type == KvCacheDataType::ZINT8)
    {
        invokeApplyBiasRopeUpdateKVCacheDispatch<T, int8_t, KVCacheBuffer, IS_GENERATE, false, true>(QKV, Q, kvTable, qkv_bias,
                                                                                                     seq_lens, kv_seq_lens, padding_offset, batch_size, seq_len, cyclic_kv_cache_len, sink_token_len, token_num,
                                                                                                     head_num, kv_head_num, size_per_head, rotary_embedding_dim, rotary_embedding_base, rotary_scale_type,
                                                                                                     rotary_embedding_scale, rotary_embedding_max_positions, position_embedding_type, medusa_position_offsets,
                                                                                                     position_shift_enabled, scale,
                                                                                                     int8_mode, enable_paged_kv_fmha, beam_width,
                                                                                                     grid_block_cache, stream);
    }

    else
    {
        invokeApplyBiasRopeUpdateKVCacheDispatch<T, T, KVCacheBuffer, IS_GENERATE, false, false>(QKV, Q, kvTable, qkv_bias, seq_lens,
                                                                                                 kv_seq_lens, padding_offset, batch_size, seq_len, cyclic_kv_cache_len, sink_token_len, token_num, head_num,
                                                                                                 kv_head_num, size_per_head, rotary_embedding_dim, rotary_embedding_base, rotary_scale_type,
                                                                                                 rotary_embedding_scale, rotary_embedding_max_positions, position_embedding_type, medusa_position_offsets,
                                                                                                 position_shift_enabled, scale,
                                                                                                 int8_mode, enable_paged_kv_fmha, beam_width,
                                                                                                 grid_block_cache, stream);
    }
}

#define INSTANTIATE_ADDFUSEDQKVBIAS_TRANSPOSE(T, KVCacheBuffer, IS_GENERATE)                                                                                                         \
    template void invokeApplyBiasRopeUpdateKVCache<T, KVCacheBuffer, IS_GENERATE>(T * QKV, T * Q,                                                                                    \
                                                                                  KVCacheBuffer & kvTable, const T *qkv_bias, const int *seq_lens, const int *kv_seq_lens,           \
                                                                                  const int *padding_offset, const int batch_size, const int seq_len, const int cyclic_kv_cache_len, \
                                                                                  const int sink_token_len, const int token_num, const int head_num, const int kv_head_num,          \
                                                                                  const int size_per_head, const int rotary_embedding_dim, const float rotary_embedding_base,        \
                                                                                  const RotaryScalingType rotary_scale_type, const float rotary_embedding_scale,                     \
                                                                                  const int rotary_embedding_max_positions, const PositionEmbeddingType position_embedding_type,     \
                                                                                  const int *medusa_position_offsets, const bool position_shift_enabled, const float *scale,         \
                                                                                  const int int8_mode, const KvCacheDataType cache_type,                                             \
                                                                                  const bool enable_paged_kv_fmha, const int beam_width, int2 &grid_block_cache, cudaStream_t stream)