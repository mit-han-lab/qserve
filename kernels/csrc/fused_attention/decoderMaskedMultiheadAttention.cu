// Inspired by TRT-LLM.
// Modified by Haotian Tang and Shang Yang.
// @article{lin2024qserve,
//   title={QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving},
//   author={Lin*, Yujun and Tang*, Haotian and Yang*, Shang and Zhang, Zhekai and Xiao, Guangxuan and Gan, Chuang and Han, Song},
//   journal={arXiv preprint arXiv:2405.04532},
//   year={2024}
// }
/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "decoderMaskedMultiheadAttention.h"
#include <assert.h>
#include <cuda_runtime.h>
#include <float.h>
#include <type_traits>
#include "decoderMaskedMultiheadAttentionTemplate.hpp"

namespace mmha
{
////////////////////////////////////////////////////////////////////////////////////////////////////


// Forward declaration of the kernel launcher to avoid including decoderMaskedMultiheadAttentionLaunch.h
template <typename T, int Dh>
inline size_t multi_block_grid_setup(const Multihead_attention_params<T>& params,
    int threads_per_block, int tlength, bool do_multi_block)
{
    if (!do_multi_block)
    {
        return 1;
    }

    auto constexpr threads_per_value = mmha::threads_per_value<T>(mmha::dh_max(Dh));

    // Make sure: seq_len_tile * threads_per_value <= threads_per_block (for multi_block_mode)
    params.seq_len_tile = std::floor(threads_per_block / threads_per_value);

    assert(params.seq_len_tile <= params.max_seq_len_tile);

    params.timesteps_per_block = mmha::divUp(tlength, params.seq_len_tile);

#ifndef ENABLE_MULTI_BLOCK_OPTION
    do_multi_block = false;
#endif

    // Return the sequence length tile if using multi block modes.
    return params.seq_len_tile;
}


#define MMHA_LAUNCH_CHECK(DYNAMIC_THDS_PER_BLOCK)                                                                      \
    std::size_t const dynamic_smem_sz{                                                                                 \
        mmha::smem_size_in_bytes<T, Dh, DO_MULTI_BLOCK>(params, DYNAMIC_THDS_PER_BLOCK)};                              \
    /* Set 46KB threshold here because we have to take static/driver shared memory into consideration. */              \
    if (dynamic_smem_sz >= 46 * 1024)                                                                                  \
    {                                                                                                                  \
        cudaError_t res = cudaFuncSetAttribute(mmha::masked_multihead_attention_kernel<T, T_cache, KVCacheBuffer, Dh,  \
                                                   DYNAMIC_THDS_PER_BLOCK, DO_MULTI_BLOCK, INT4KV, KV_WITH_ZEROS, SMEM_PRELOAD>,                 \
            cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_smem_sz);                                             \
    }                                                                                                                  \
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&available_blocks,                                                   \
        mmha::masked_multihead_attention_kernel<T, T_cache, KVCacheBuffer, Dh, DYNAMIC_THDS_PER_BLOCK,      \
            DO_MULTI_BLOCK, INT4KV, KV_WITH_ZEROS, SMEM_PRELOAD>,                                                                                           \
        DYNAMIC_THDS_PER_BLOCK, dynamic_smem_sz);


#define MMHA_KERNEL(DYNAMIC_THDS_PER_BLOCK)                                                                            \
    std::size_t const dynamic_smem_sz{                                                                                 \
        mmha::smem_size_in_bytes<T, Dh, DO_MULTI_BLOCK>(params, DYNAMIC_THDS_PER_BLOCK)};                              \
    /* Set 46KB threshold here because we have to take static/driver shared memory into consideration. */              \
    if (dynamic_smem_sz >= 46 * 1024)                                                                                  \
    {                                                                                                                  \
        cudaError_t res = cudaFuncSetAttribute(                                                                        \
            mmha::masked_multihead_attention_kernel<T, T_cache, KVCacheBuffer, Dh, DYNAMIC_THDS_PER_BLOCK,             \
                 DO_MULTI_BLOCK, INT4KV, KV_WITH_ZEROS, SMEM_PRELOAD>,                                      \
            cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_smem_sz);                                             \
    }                                                                                                                  \
    mmha::masked_multihead_attention_kernel<T, T_cache, KVCacheBuffer, Dh, DYNAMIC_THDS_PER_BLOCK,                     \
         DO_MULTI_BLOCK, INT4KV, KV_WITH_ZEROS, SMEM_PRELOAD>                                               \
        <<<grid, DYNAMIC_THDS_PER_BLOCK, dynamic_smem_sz, stream>>>(params, kv_cache_buffer);


// if resources are not enough to launch 512 threads per block, we will fallback to 256.
#define MMHA_LAUNCH_512_BLOCKSIZE()                                                                                    \
    int available_blocks = -1;                                                                                         \
    MMHA_LAUNCH_CHECK(512);                                                                                            \
    if (available_blocks <= 0)                                                                                         \
    {                                                                                                                  \
        MMHA_KERNEL(256);                                                                                              \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        MMHA_KERNEL(512);                                                                                              \
    }

// if resources are not enough to launch 1024 threads per block, we will fallback to 512.
#define MMHA_LAUNCH_1024_BLOCKSIZE()                                                                                   \
    int available_blocks = -1;                                                                                         \
    MMHA_LAUNCH_CHECK(1024);                                                                                           \
    if (available_blocks <= 0)                                                                                         \
    {                                                                                                                  \
        MMHA_LAUNCH_512_BLOCKSIZE();                                                                                   \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        MMHA_KERNEL(1024);                                                                                             \
    }

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, typename T_cache, typename KVCacheBuffer, typename KernelParamsType, int Dh, int THDS_PER_BLOCK,
    bool DO_MULTI_BLOCK, bool INT4KV, bool KV_WITH_ZEROS, bool SMEM_PRELOAD>
void mmha_launch_kernel_ex(
    const KernelParamsType& params, const KVCacheBuffer& kv_cache_buffer, const cudaStream_t& stream, int tlength)
{
    std::size_t const seq_len_tile{mmha::multi_block_grid_setup<T, Dh>(
        params, THDS_PER_BLOCK, tlength, DO_MULTI_BLOCK)};
    dim3 grid{static_cast<unsigned>(params.num_heads), static_cast<unsigned>(params.batch_size),
        static_cast<unsigned>(seq_len_tile)};

    if (DO_MULTI_BLOCK)
    {
        MMHA_KERNEL(THDS_PER_BLOCK);
    }
    else
    {
        const int kernel_total_blocks = params.batch_size * params.num_heads;
        // Don't tune the block size if batchxhead is large enough.
        // The max number of warps we can launch per SM is 32 limited by registers.
        if (kernel_total_blocks >= params.multi_processor_count * 4)
        {
            MMHA_KERNEL(THDS_PER_BLOCK);
            return;
        }

        // Tune block size based on batchxhead to increase occupancy.
        int num_blocks_per_sm = -1;
        // Set 0 dynamic shared memory size as we need the number of available blocks limited by registers.
        // Dynamic shared memory is fixed for different block size.
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm,
            mmha::masked_multihead_attention_kernel<T, T_cache, KVCacheBuffer, Dh, THDS_PER_BLOCK,
                DO_MULTI_BLOCK, INT4KV, KV_WITH_ZEROS, SMEM_PRELOAD>,
            THDS_PER_BLOCK, 0);

        int block_size_factor = min(
            mmha::divUp(params.multi_processor_count * num_blocks_per_sm, kernel_total_blocks), num_blocks_per_sm);
        // Max block size is 1024.
        const int dynamic_block_size = min(THDS_PER_BLOCK * block_size_factor, 1024);

        // Make sure number of threads per block is power of 2.
        if (dynamic_block_size <= 256)
        {
            MMHA_KERNEL(256);
        }
        else if (dynamic_block_size <= 512)
        {
            // Check if the kernel with new block size can be launched in terms of resources.
            MMHA_LAUNCH_512_BLOCKSIZE();
        }
        else if (dynamic_block_size <= 1024)
        {
            // Check if the kernel with new block size can be launched in terms of resources.
            MMHA_LAUNCH_1024_BLOCKSIZE();
        }
    }
}

template <typename T, typename KVCacheBuffer, typename KernelParamsType, int Dh, int THDS_PER_BLOCK,
    bool DO_MULTI_BLOCK>
void mmha_launch_kernel_dispatch_4bits_kv_cache(
    const KernelParamsType& params, const KVCacheBuffer& kv_cache_buffer, const cudaStream_t& stream, int tlength)
{
    if (params.int4_kv_cache)
    {
        if (params.kv_cache_with_zeros)
        {
            if (params.timestep < 2048)
            {
                // Note: the 4bit kv_cache is still packed in int8_t.
                mmha_launch_kernel_ex<T, int8_t, KVCacheBuffer, KernelParamsType, Dh, THDS_PER_BLOCK,
                    DO_MULTI_BLOCK, true, true, true>(params, kv_cache_buffer, stream, tlength);
            }
            else
            {
                // Note: the 4bit kv_cache is still packed in int8_t.
                mmha_launch_kernel_ex<T, int8_t, KVCacheBuffer, KernelParamsType, Dh, THDS_PER_BLOCK,
                    DO_MULTI_BLOCK, true, true, false>(params, kv_cache_buffer, stream, tlength);
            }
        }
        else
        {
            if (params.timestep < 2048)
            {
                // Note: the 4bit kv_cache is still packed in int8_t.
                mmha_launch_kernel_ex<T, int8_t, KVCacheBuffer, KernelParamsType, Dh, THDS_PER_BLOCK,
                    DO_MULTI_BLOCK, true, false, true>(params, kv_cache_buffer, stream, tlength);
            }
            else
            {
                // Note: the 4bit kv_cache is still packed in int8_t.
                mmha_launch_kernel_ex<T, int8_t, KVCacheBuffer, KernelParamsType, Dh, THDS_PER_BLOCK,
                    DO_MULTI_BLOCK, true, false, false>(params, kv_cache_buffer, stream, tlength);
            }

        }
    }
    else
    {
        // this should never happen
        // mmha_launch_kernel_ex<T, T, KVCacheBuffer, KernelParamsType, Dh, THDS_PER_BLOCK, DO_MULTI_BLOCK, true>(
        //     params, kv_cache_buffer, stream, tlength);
    }
}

template <typename T, typename KVCacheBuffer, typename KernelParamsType, int Dh, int THDS_PER_BLOCK,
    bool DO_MULTI_BLOCK>
void mmha_launch_kernel_dispatch_8bits_kv_cache(
    const KernelParamsType& params, const KVCacheBuffer& kv_cache_buffer, const cudaStream_t& stream, int tlength)
{
    if (params.int8_kv_cache)
    {
        if (params.kv_cache_with_zeros)
        {
            mmha_launch_kernel_ex<T, int8_t, KVCacheBuffer, KernelParamsType, Dh, THDS_PER_BLOCK,
                DO_MULTI_BLOCK, false, true, false>(params, kv_cache_buffer, stream, tlength);
        }
        else
        {
            mmha_launch_kernel_ex<T, int8_t, KVCacheBuffer, KernelParamsType, Dh, THDS_PER_BLOCK,
                DO_MULTI_BLOCK, false, false, false>(params, kv_cache_buffer, stream, tlength);
        }
    }
#ifdef ENABLE_FP8
    else if (params.fp8_kv_cache)
    {
        mmha_launch_kernel_ex<T, __nv_fp8_e4m3, KVCacheBuffer, KernelParamsType, Dh, THDS_PER_BLOCK,
            DO_MULTI_BLOCK, false, false, false>(params, kv_cache_buffer, stream, tlength);
    }
#endif // ENABLE_FP8
    else
    {
        mmha_launch_kernel_ex<T, T, KVCacheBuffer, KernelParamsType, Dh, THDS_PER_BLOCK, DO_MULTI_BLOCK, false, false, false>(
            params, kv_cache_buffer, stream, tlength);
    }
}

template <typename T, typename KVCacheBuffer, typename KernelParamsType, int Dh>
void mmha_launch_kernel_dispatch(
    const KernelParamsType& params, const KVCacheBuffer& kv_cache_buffer, const cudaStream_t& stream)
{
    int const tlength = params.timestep;
    bool int4_kv_cache = params.int4_kv_cache;
    if (int4_kv_cache)
    {
        if (tlength < 1024)
        {
            mmha_launch_kernel_dispatch_4bits_kv_cache<T, KVCacheBuffer, KernelParamsType, Dh, 256, false>(
                params, kv_cache_buffer, stream, tlength);
        }
        else
        {
            if (params.multi_block_mode)
            {
                mmha_launch_kernel_dispatch_4bits_kv_cache<T, KVCacheBuffer, KernelParamsType, Dh, 256, true>(
                    params, kv_cache_buffer, stream, tlength);
            }
            else
            {
                mmha_launch_kernel_dispatch_4bits_kv_cache<T, KVCacheBuffer, KernelParamsType, Dh, 256, false>(
                    params, kv_cache_buffer, stream, tlength);
            }
        }
    }
    else    // int8_kv_cache
    {
        if (tlength < 1024)
        {
            mmha_launch_kernel_dispatch_8bits_kv_cache<T, KVCacheBuffer, KernelParamsType, Dh, 256, false>(
                params, kv_cache_buffer, stream, tlength);
        }
        else
        {
            if (params.multi_block_mode)
            {
                mmha_launch_kernel_dispatch_8bits_kv_cache<T, KVCacheBuffer, KernelParamsType, Dh, 256, true>(
                    params, kv_cache_buffer, stream, tlength);
            }
            else
            {
                mmha_launch_kernel_dispatch_8bits_kv_cache<T, KVCacheBuffer, KernelParamsType, Dh, 256, false>(
                    params, kv_cache_buffer, stream, tlength);
            }
        }
    }
}

template <typename T, typename KVCacheBuffer, typename KernelParamsType, int Dh>
void mmha_launch_kernel(
    const KernelParamsType& params, const KVCacheBuffer& kv_cache_buffer, const cudaStream_t& stream)
{
    // assert((params.rotary_embedding_dim != 0)
    //     == (params.position_embedding_type == PositionEmbeddingType::kROPE_GPT_NEOX
    //         || params.position_embedding_type == PositionEmbeddingType::kROPE_GPTJ));
    // if (params.beam_width == 1)
    // {
    //     mmha_launch_kernel_dispatch<T, KVCacheBuffer, KernelParamsType, Dh, false>(params, kv_cache_buffer, stream);
    // }
    // else
    // {
    //     mmha_launch_kernel_dispatch<T, KVCacheBuffer, KernelParamsType, Dh, true>(params, kv_cache_buffer, stream);
    // }
    mmha_launch_kernel_dispatch<T, KVCacheBuffer, KernelParamsType, Dh>(params, kv_cache_buffer, stream);
}

} // namespace mmha

namespace
{

#define MMHA_LAUNCH_KERNEL(Dh)                                                                                         \
    mmha::mmha_launch_kernel<T, KVCacheBuffer, KERNEL_PARAMS_TYPE, Dh>(params, kv_cache_buffer, stream);               \
    break;

template <typename T, typename KVCacheBuffer, typename KERNEL_PARAMS_TYPE>
void multihead_attention_(
    const KERNEL_PARAMS_TYPE& params, const KVCacheBuffer& kv_cache_buffer, const cudaStream_t& stream)
{
    switch (params.hidden_size_per_head)
    {
    // case 32: mmha::mmha_launch_kernel<T, KVCacheBuffer, KERNEL_PARAMS_TYPE, 32>(params, kv_cache_buffer, stream); break;
    // case 48: mmha::mmha_launch_kernel<T, KVCacheBuffer, KERNEL_PARAMS_TYPE, 48>(params, kv_cache_buffer, stream); break;
    // case 64: mmha::mmha_launch_kernel<T, KVCacheBuffer, KERNEL_PARAMS_TYPE, 64>(params, kv_cache_buffer, stream); break;
    // case 80: mmha::mmha_launch_kernel<T, KVCacheBuffer, KERNEL_PARAMS_TYPE, 80>(params, kv_cache_buffer, stream); break;
    // case 96: mmha::mmha_launch_kernel<T, KVCacheBuffer, KERNEL_PARAMS_TYPE, 96>(params, kv_cache_buffer, stream); break;
    // case 112:
    //     mmha::mmha_launch_kernel<T, KVCacheBuffer, KERNEL_PARAMS_TYPE, 112>(params, kv_cache_buffer, stream);
    //     break;
    case 128:
        mmha::mmha_launch_kernel<T, KVCacheBuffer, KERNEL_PARAMS_TYPE, 128>(params, kv_cache_buffer, stream);
        break;
    // case 144:
    //     mmha::mmha_launch_kernel<T, KVCacheBuffer, KERNEL_PARAMS_TYPE, 144>(params, kv_cache_buffer, stream);
    //     break;
    // case 160:
    //     mmha::mmha_launch_kernel<T, KVCacheBuffer, KERNEL_PARAMS_TYPE, 160>(params, kv_cache_buffer, stream);
    //     break;
    // case 192:
    //     mmha::mmha_launch_kernel<T, KVCacheBuffer, KERNEL_PARAMS_TYPE, 192>(params, kv_cache_buffer, stream);
    //     break;
    // case 224:
    //     mmha::mmha_launch_kernel<T, KVCacheBuffer, KERNEL_PARAMS_TYPE, 224>(params, kv_cache_buffer, stream);
    //     break;
    // case 256:
    //     mmha::mmha_launch_kernel<T, KVCacheBuffer, KERNEL_PARAMS_TYPE, 256>(params, kv_cache_buffer, stream);
    //     break;
    default: assert(false);
    }
}

#undef MMHA_LAUNCH_KERNEL

} // namespace

////////////////////////////////////////////////////////////////////////////////////////////////////

#define INSTANTIATE_MMHA_NORMAL_AND_PAGED(T)                                                          \
    void masked_multihead_attention(const Multihead_attention_params<T>& params,                      \
        const KVBlockArray& kv_cache_buffer, const cudaStream_t& stream)                                               \
    {                                                                                                                  \
        multihead_attention_<T, KVBlockArray, Multihead_attention_params<T>>(                         \
            params, kv_cache_buffer, stream);                                                                          \
    }                                                                                                                  \
    void masked_multihead_attention(const Multihead_attention_params<T>& params,                      \
        const KVLinearBuffer& kv_cache_buffer, const cudaStream_t& stream)                                             \
    {                                                                                                                  \
        multihead_attention_<T, KVLinearBuffer, Multihead_attention_params<T>>(                       \
            params, kv_cache_buffer, stream);                                                                          \
    }
//INSTANTIATE_MMHA_NORMAL_AND_PAGED(float, true)
// INSTANTIATE_MMHA_NORMAL_AND_PAGED(float)
//INSTANTIATE_MMHA_NORMAL_AND_PAGED(uint16_t, true)
INSTANTIATE_MMHA_NORMAL_AND_PAGED(uint16_t)
#undef INSTANTIATE_MMHA_NORMAL_AND_PAGED

////////////////////////////////////////////////////////////////////////////////////////////////////

