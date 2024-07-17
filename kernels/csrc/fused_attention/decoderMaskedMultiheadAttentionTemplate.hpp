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
#pragma once

#include "cudaTypeUtils.cuh"
#include "memoryUtils.h"
#include "decoderMaskedMultiheadAttention.h"
#include "decoderMaskedMultiheadAttentionUtils.h"
#include "kvCacheUtils.h"
#include <cuda_fp16.h>
#include <cuda_pipeline_primitives.h>
#include <assert.h>
#include <float.h>
#include <type_traits>

// Multi-block mmha kernel can only be selected when CUDA >= 11.7
#if (CUDART_VERSION >= 11070)
#define ENABLE_MULTI_BLOCK_OPTION
#endif

#if (__CUDACC_VER_MAJOR__ >= 11) && (__CUDACC_VER_MINOR__ >= 4)
#define L2_CACHEHINT(size) ".L2::" #size "B"
#else
#define L2_CACHEHINT(size)
#endif

// #ifdef ENABLE_MULTI_BLOCK_OPTION
// #include <cub/block/block_reduce.cuh>
// #include <cuda/atomic>
// #include <cuda/std/bit>
// #endif // ENABLE_MULTI_BLOCK_OPTION

// #define MMHA_USE_HMMA_FOR_REDUCTION

// Below are knobs to extend FP32 accumulation for higher FP16 accuracy

// Does not seem to affect the accuracy that much
#define MMHA_USE_FP32_ACUM_FOR_FMA

// Seems to slightly improve the accuracy
#define MMHA_USE_FP32_ACUM_FOR_OUT

#if 0 && defined(MMHA_USE_FP32_ACUM_FOR_OUT)
 // Does not seem to improve the accuracy
 //#define MMHA_USE_FP32_ACUM_FOR_LOGITS
#endif

namespace mmha
{

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    //
    // We use the following terminology to describe the different dimensions.
    //
    // B:  Batch size (number of sequences),
    // L:  Sequence length,
    // D:  Hidden dimension,
    // H:  Number of heads,
    // Dh: Hidden dimension per head - Dh = D / H.
    //
    // The different kernels assign a threadblock for B x H pair. The grid has size (1, B, H). We use
    // 256 threads per block to maximum occupancy and performance.
    //
    // Each threadblock loads Dh values from Q and its associated bias. The kernels run a loop to
    // compute Q * K^T where K is loaded from a cache buffer -- except for the current timestep. The
    // cache buffer helps with memory accesses and contains keys with bias.
    //
    // The layout of the cache buffer for the keys/values is [B, H, L, Dh]
    // where the fastest moving dimension (contiguous data) is the rightmost one.
    // Contiguous threads will read one hidden_dimension per LDG unless we need more than 32 threads.
    //
    // The different kernels use 1 ~ 32 threads per key (THREADS_PER_KEY). The size of the LDGs
    // is always 16bytes (8 bytes for 8bit cache). Each thread sums Dh / THREADS_PER_KEY elements. At
    // the end of each iteration of the Q * K^T loop, we perform a reduction between lanes using an
    // HMMA instruction (Tensor Core). Each Q * K^T value is stored in shared memory in FP32.
    //
    // After that loop, a parallel softmax is computed across the different Q * K^T values stored in
    // shared memory.
    //
    // The kernel ends with a loop over the values in V. We use THREADS_PER_VALUE to control how many
    // timesteps are computed by loop iteration. As with the keys, the values are read from a cache
    // except for the current timestep. The layout of the cache buffer for the values is same as the key,
    // which is [B, H, L, Dh].
    //
    // Note that we have remapped key layout to make sure it shares the same pattern as value [B, H, L, Dh].
    // It helps coalescing memory access, and reducing register pressure.

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    template <typename T, int Dh_MAX>
    struct Qk_vec_m_
    {
    };

    template <>
    struct Qk_vec_m_<uint16_t, 32>
    {
        using Type = uint32_t;
    };

    template <>
    struct Qk_vec_m_<uint16_t, 64>
    {
        using Type = uint32_t;
    };

    template <>
    struct Qk_vec_m_<uint16_t, 128>
    {
        using Type = uint2;
    };

    template <>
    struct Qk_vec_m_<uint16_t, 256>
    {
        using Type = uint4;
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <typename T, int Dh>
    struct Qk_vec_k_
    {
        using Type = typename Qk_vec_m_<T, Dh>::Type;
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <typename T, int V_VEC_SIZE>
    struct V_vec_m_
    {
    };

    template <>
    struct V_vec_m_<uint16_t, 2>
    {
        using Type = uint32_t;
    };

    template <>
    struct V_vec_m_<uint16_t, 4>
    {
        using Type = uint2;
    };

    template <>
    struct V_vec_m_<uint16_t, 8>
    {
        using Type = uint4;
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <typename T, int V_VEC_SIZE>
    struct V_vec_k_
    {
        using Type = typename V_vec_m_<T, V_VEC_SIZE>::Type;
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    // Reuse V_vec traits as key and value share the same layout.
    template <typename T, int K_VEC_SIZE>
    struct K_vec_m_
    {
        using Type = typename V_vec_m_<T, K_VEC_SIZE>::Type;
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <typename T, int K_VEC_SIZE>
    struct K_vec_k_
    {
        using Type = typename K_vec_m_<T, K_VEC_SIZE>::Type;
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef MMHA_USE_FP32_ACUM_FOR_FMA
    template <typename T>
    struct Qk_vec_acum_fp32_
    {
    };

    template <>
    struct Qk_vec_acum_fp32_<float>
    {
        using Type = float;
    };

    template <>
    struct Qk_vec_acum_fp32_<float2>
    {
        using Type = float2;
    };

    template <>
    struct Qk_vec_acum_fp32_<float4>
    {
        using Type = float4;
    };

    // template<> struct Qk_vec_acum_fp32_<uint16_t> { using Type = float;        };
    template <>
    struct Qk_vec_acum_fp32_<uint32_t>
    {
        using Type = float2;
    };

    template <>
    struct Qk_vec_acum_fp32_<uint2>
    {
        using Type = Float4_;
    };

    template <>
    struct Qk_vec_acum_fp32_<uint4>
    {
        using Type = Float8_;
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <typename T>
    struct K_vec_acum_fp32_
    {
    };

    template <>
    struct K_vec_acum_fp32_<float>
    {
        using Type = float;
    };

    template <>
    struct K_vec_acum_fp32_<float2>
    {
        using Type = float2;
    };

    template <>
    struct K_vec_acum_fp32_<float4>
    {
        using Type = float4;
    };

    template <>
    struct K_vec_acum_fp32_<Float8_>
    {
        using Type = Float8_;
    };

    template <>
    struct K_vec_acum_fp32_<uint32_t>
    {
        using Type = float2;
    };

    template <>
    struct K_vec_acum_fp32_<uint2>
    {
        using Type = Float4_;
    };

    template <>
    struct K_vec_acum_fp32_<uint4>
    {
        using Type = Float8_;
    };

#endif // MMHA_USE_FP32_ACUM_FOR_FMA

    ////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef MMHA_USE_FP32_ACUM_FOR_OUT
    template <typename T>
    struct V_vec_acum_fp32_
    {
    };

    template <>
    struct V_vec_acum_fp32_<float>
    {
        using Type = float;
    };

    template <>
    struct V_vec_acum_fp32_<float2>
    {
        using Type = float2;
    };

    template <>
    struct V_vec_acum_fp32_<float4>
    {
        using Type = float4;
    };

    template <>
    struct V_vec_acum_fp32_<uint32_t>
    {
        using Type = float2;
    };

    template <>
    struct V_vec_acum_fp32_<uint2>
    {
        using Type = Float4_;
    };

    template <>
    struct V_vec_acum_fp32_<uint4>
    {
        using Type = Float8_;
    };
#endif

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <typename Tout, typename Tin>
    __inline__ __device__ constexpr Tout vec_conversion(const Tin &x)
    {
        static_assert(std::is_same<Tout, Tin>::value, "Type mismatch");
        return x;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <int THREADS_PER_KEY, typename K_vec, int N>
    inline __device__ float qk_dot_(const K_vec (&q)[N], const K_vec (&k)[N])
    {
#ifdef MMHA_USE_FP32_ACUM_FOR_FMA
        using K_vec_acum = typename K_vec_acum_fp32_<K_vec>::Type;
#else
        using K_vec_acum = K_vec;
#endif
        // Compute the parallel products for Q*K^T (treat vector lanes separately).
        K_vec_acum qk_vec = mul<K_vec_acum, K_vec, K_vec>(q[0], k[0]);
#pragma unroll
        for (int ii = 1; ii < N; ++ii)
        {
            qk_vec = fma(q[ii], k[ii], qk_vec);
        }

        // Finalize the reduction across lanes.
        float qk = sum(qk_vec);
#pragma unroll
        for (int mask = THREADS_PER_KEY / 2; mask >= 1; mask /= 2)
        {
            qk += __shfl_xor_sync(uint32_t(-1), qk, mask);
        }
        return qk;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <typename T, int THREADS_PER_KEY>
    struct Qk_dot
    {
        template <typename K_vec, int N>
        static inline __device__ float dot(const K_vec (&q)[N], const K_vec (&k)[N])
        {
            return qk_dot_<THREADS_PER_KEY>(q, k);
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ float4 hmma_fp32(const uint2 &a, uint32_t b)
    {
        float4 c;
        float zero = 0.f;
        asm volatile(
            "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 \n"
            "    {%0, %1, %2, %3}, \n"
            "    {%4, %5}, \n"
            "    {%6}, \n"
            "    {%7, %7, %7, %7}; \n"

            : "=f"(c.x), "=f"(c.y), "=f"(c.z), "=f"(c.w)
            : "r"(a.x) "r"(a.y), "r"(b), "f"(zero));
        return c;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <int N>
    inline __device__ float qk_hmma_dot_(const uint32_t (&q)[N], const uint32_t (&k)[N])
    {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
#ifdef MMHA_USE_FP32_ACUM_FOR_FMA
        using K_vec_acum = typename K_vec_acum_fp32_<uint32_t>::Type;
#else
        using K_vec_acum = uint32_t;
#endif
        K_vec_acum qk_vec = mul<K_vec_acum, uint32_t, uint32_t>(q[0], k[0]);
#pragma unroll
        for (int ii = 1; ii < N; ++ii)
        {
            qk_vec = fma(q[ii], k[ii], qk_vec);
        }
#ifdef MMHA_USE_FP32_ACUM_FOR_FMA
        uint32_t qk_vec_ = float2_to_half2(qk_vec);
        return hmma_fp32(make_uint2(qk_vec_, 0u), 0x3c003c00u).x;
#else
        return hmma_fp32(make_uint2(qk_vec, 0u), 0x3c003c00u).x;
#endif
#else
        return 0.f;
#endif
    }


    template <int THREADS_PER_KEY, typename K_vec_k>
    inline __device__ float qk_hmma_dot_simple(const K_vec_k& q, const K_vec_k& k);

    template <int THREADS_PER_KEY>
    inline __device__ float qk_hmma_dot_simple(const uint32_t& q, const uint32_t& k)
    {
        assert (0);
    }

    template <int THREADS_PER_KEY>
    inline __device__ float qk_hmma_dot_simple(const uint2& q, const uint2& k)
    {
        assert (0);
    }

    template <int THREADS_PER_KEY>
    inline __device__ float qk_hmma_dot_simple(const uint4& q, const uint4& k)
    {
        using K_vec_acum = uint32_t;
        K_vec_acum qk_vec = mul<K_vec_acum, uint32_t, uint32_t>(q.x, k.x);
        asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(qk_vec) : "r"(q.y), "r"(k.y), "r"(qk_vec));
        asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(qk_vec) : "r"(q.z), "r"(k.z), "r"(qk_vec));
        asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(qk_vec) : "r"(q.w), "r"(k.w), "r"(qk_vec));
        // return hmma_fp32(make_uint2(qk_vec, 0u), 0x3c003c00u).x;
        half2 qk_vec_h = (half2 &)qk_vec;
        float qk = __half2float(__hadd(qk_vec_h.x, qk_vec_h.y));
#pragma unroll
        for (int mask = THREADS_PER_KEY / 2; mask >= 1; mask /= 2)
        {
            qk += __shfl_xor_sync(uint32_t(-1), qk, mask);
        }
        return qk;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <>
    struct Qk_dot<uint16_t, 4>
    {
        template <typename K_vec, int N>
        static inline __device__ float dot(const K_vec (&q)[N], const K_vec (&k)[N])
        {
            return qk_dot_<4>(q, k);
        }

        template <int N>
        static inline __device__ float dot(const uint32_t (&q)[N], const uint32_t (&k)[N])
        {
#if __CUDA_ARCH__ >= 750 && defined(MMHA_USE_HMMA_FOR_REDUCTION)
            return qk_hmma_dot_(q, k);
#else
            return qk_dot_<4>(q, k);
#endif // defined MMHA_USE_HMMA_FOR_REDUCTION
        }
    };


    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <int WARPS_PER_BLOCK, int WARP_SIZE = 32>
    inline __device__ float block_sum(float *red_smem, float sum)
    {

        // Decompose the thread index into warp / lane.
        int warp = threadIdx.x / WARP_SIZE;
        int lane = threadIdx.x % WARP_SIZE;

// Compute the sum per warp.
#pragma unroll
        for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2)
        {
            sum += __shfl_xor_sync(uint32_t(-1), sum, mask);
        }

        // Warp leaders store the data to shared memory.
        if (lane == 0)
        {
            red_smem[warp] = sum;
        }

        // Make sure the data is in shared memory.
        __syncthreads();

        // The warps compute the final sums.
        if (lane < WARPS_PER_BLOCK)
        {
            sum = red_smem[lane];
        }

// Parallel reduction inside the warp.
#pragma unroll
        for (int mask = WARPS_PER_BLOCK / 2; mask >= 1; mask /= 2)
        {
            sum += __shfl_xor_sync(uint32_t(-1), sum, mask);
        }

        // Broadcast to other threads.
        return __shfl_sync(uint32_t(-1), sum, 0);
    }

#if defined(MMHA_USE_FP32_ACUM_FOR_LOGITS)

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ float cast_to_float(float u)
    {
        return u;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ float2 cast_to_float(float2 u)
    {
        return u;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ float4 cast_to_float(float4 u)
    {
        return u;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ Float4_ cast_to_float(Float4_ u)
    {
        return u;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ Float8_ cast_to_float(Float8_ u)
    {
        return u;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ float2 cast_to_float(uint32_t u)
    {
        return half2_to_float2(u);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ Float4_ cast_to_float(uint2 u)
    {
        Float4_ tmp;
        tmp.x = half2_to_float2(u.x);
        tmp.y = half2_to_float2(u.y);
        return tmp;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ Float8_ cast_to_float(uint4 u)
    {
        Float8_ tmp;
        tmp.x = half2_to_float2(u.x);
        tmp.y = half2_to_float2(u.y);
        tmp.z = half2_to_float2(u.z);
        tmp.w = half2_to_float2(u.w);
        return tmp;
    }

#endif

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <typename T>
    inline __device__ __host__ T divUp(T m, T n)
    {
        return (m + n - 1) / n;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <typename T>
    inline __device__ __host__ T div(T m, T n)
    {
        return m / n;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <typename T>
    struct kernel_type_t
    {
        using Type = T;
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    // Compute the largest supported head size (dh_max). It must be the smallest power-of-2 that is not strictly smaller
    // than the head size (dh).
    inline __device__ __host__ constexpr unsigned dh_max(unsigned dh)
    {
        return next_power_of_two(const_max(dh, 32u));
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <typename T>
    inline __device__ __host__ constexpr unsigned threads_per_value(unsigned dh_max)
    {
        return dh_max * sizeof(T) / 16;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <typename T, unsigned Dh_MAX>
    inline __device__ __host__ constexpr unsigned threads_per_key()
    {
        // Since we want to perform the reduction entirely within a warp, the number of threads per key
        // is capped at 32.
        constexpr unsigned threads = (unsigned)(Dh_MAX * sizeof(T) / 16u);
        if ((threads & (threads - 1)) != 0)
        {
            assert(false); // Not a power of two.
        }
        return std::min(32u, threads);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <typename T, typename T_VEC, unsigned VECS_PER_CHUNK>
    __device__ inline constexpr uint2 chunk_index(unsigned tidx)
    {
        // The chunk associated with the thread.
        auto const idx_chunk = tidx / VECS_PER_CHUNK;

        // The position of the T_VEC vector in that chunk associated with the thread.
        static_assert(sizeof(T_VEC) % sizeof(T) == 0);
        unsigned constexpr kVecSize{sizeof(T_VEC) / sizeof(T)};
        auto const idx_vec = (tidx % VECS_PER_CHUNK) * kVecSize;

        return uint2{idx_chunk, idx_vec};
    }


    ////////////////////////////////////////////////////////////////////////////////////////////////////
    __inline__ __device__ uint32_t cast_smem_ptr_to_uint_helper(void const *const ptr)
    {
        uint32_t smem_int_ptr;

        asm("{.reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, "
            "smem_ptr; }\n"
            : "=r"(smem_int_ptr)
            : "l"(ptr));

        return smem_int_ptr;
    }

    __inline__ __device__ void
    cp_async_helper(uint32_t smem_int_ptr, const uint4 *__restrict__ src, bool mask)
    {
        const int cp_size = 16;
        // cachehint will not impact performance.
        // clang-format off
        asm volatile("{"
                        "  .reg .pred p;"
                        "  setp.ne.b32 p, %0, 0;"
                        "  @p cp.async.cg.shared.global" L2_CACHEHINT(128) " [%1], [%2], %3;"
                        "}" ::"r"((int)mask),
                        "r"(smem_int_ptr),
                        "l"(src),
                        "n"(cp_size));
        // clang-format on
    }


    __inline__ __device__ void
    cp_async_launch(void *dst_ptr, const uint4 *__restrict__ src_ptr, bool mask)
    {
       uint32_t addr = cast_smem_ptr_to_uint_helper(dst_ptr);
       cp_async_helper(addr, src_ptr, mask);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////


    template <
        // The type of the inputs. Supported types: float, uint16_t, nv_bfloat16.
        typename T,
        // The type of the cache.
        typename Tcache,
        // Type of struct containing KV cache
        typename KVCacheBuffer,
        // The hidden dimension per head.
        unsigned Dh,
        // The number of threads in a threadblock.
        unsigned THREADS_PER_BLOCK,
        // Whether enable multi-block mode for long-sequence-length.
        bool DO_MULTI_BLOCK = false,
        // Whether use INT4KV
        bool INT4KV = false,
        bool KV_WITH_ZEROS = false,
        bool SMEM_PRELOAD = false,
        // The number of threads per key.
        unsigned THREADS_PER_KEY = mmha::threads_per_key<T, dh_max(Dh)>(),
        // The number of threads per value.
        unsigned THREADS_PER_VALUE = mmha::threads_per_value<T>(dh_max(Dh)),
        // The unroll factor for loading from K cache.
        // unsigned K_LOOP_UNROLL = 8, // 8,
        // The unroll factor for loading from V cache.
        // Set it default to 4 for higher occupancy (by reducing registers usage).
        unsigned V_LOOP_UNROLL = 4>
    __global__ void masked_multihead_attention_kernel(
        Multihead_attention_params<T> params, KVCacheBuffer kvCacheBuffer)
    {
        constexpr unsigned K_LOOP_UNROLL = SMEM_PRELOAD ? 8 : 4;
        using Tk = typename kernel_type_t<T>::Type;
        // Use 8bit cache.
        static constexpr bool ENABLE_8BITS_CACHE = sizeof(Tcache) == 1;
        static constexpr bool ENABLE_4BITS_CACHE = (INT4KV && ENABLE_8BITS_CACHE);
        static constexpr bool ENABLE_ZEROS = KV_WITH_ZEROS;

        // The size of a warp.
        constexpr unsigned WARP_SIZE{32};
        // The number of warps in a threadblock.
        constexpr unsigned WARPS_PER_BLOCK{THREADS_PER_BLOCK / WARP_SIZE};

        // The maximum hidden size per head.
        constexpr auto Dh_MAX = dh_max(Dh);
        constexpr bool IS_Dh_MAX = Dh == Dh_MAX;
        static_assert(Dh_MAX >= WARP_SIZE);
        static_assert(Dh_MAX >= Dh);

        // The maximum sequence length in the kv_cache, i.e., an upper bound on L.
        // Note that the maximum sequence length supported by the model might be greater than this.
        const auto max_seq_len = static_cast<unsigned>(params.memory_max_len);
        assert(max_seq_len > 0);
        // The current timestep (including paddings).
        // It is only used to calculate the smem stride.
        const auto timestep = static_cast<unsigned>(DO_MULTI_BLOCK ? params.timesteps_per_block : params.timestep);

        constexpr bool MULTI_BLOCK_FLAG = false;

        // Use smem_size_in_bytes (above) to determine the amount of shared memory.
        extern __shared__ char smem_[];

        // The shared memory for the Q*K^T values and partial logits in softmax.
        auto qk_smem = reinterpret_cast<float *>(smem_);

        __shared__ float qk_current_smem[1];

        // The shared memory for the logits. For FP32, that's the same buffer as qk_smem.
        char *logits_smem_ = smem_;
#ifndef MMHA_USE_FP32_ACUM_FOR_LOGITS
        if (sizeof(Tk) != 4)
        {
            // TODO - change to tlength
            const auto max_timesteps = min(timestep, max_seq_len); // const auto max_timesteps = DO_CROSS_ATTENTION ? max_seq_len : min(timestep, max_seq_len);
            logits_smem_ += divUp(max_timesteps + 1, 4u) * 16;
        }
        Tk *logits_smem = reinterpret_cast<Tk *>(logits_smem_);
#else
        float *logits_smem = reinterpret_cast<float *>(logits_smem_);
#endif

        __shared__ Tk logits_current_smem[1];

        // The shared memory to do the final reduction for the output values. Reuse qk_smem.
        Tk *out_smem = reinterpret_cast<Tk *>(smem_);

        // The shared memory buffers for the block-wide reductions. One for max, one for sum.
        __shared__ float red_smem[WARPS_PER_BLOCK * 2];

        // A vector of Q or K elements for the current timestep.
        using Qk_vec_m = typename Qk_vec_m_<T, Dh_MAX>::Type; // with memory-used precision
        using Qk_vec_k = typename Qk_vec_k_<T, Dh_MAX>::Type; // with kernel-used precision

        // Make sure the hidden dimension per head is a multiple of the number of threads per key.
        static_assert(Dh_MAX % THREADS_PER_KEY == 0); // trivially satisfied since THREADS_PER_KEY in {1, 2, 4}

        // The number of elements per vector.
        // Each thread will handle 16 bytes.
        constexpr int K_VEC_SIZE = 16u / sizeof(T);
        // Make sure the hidden size per head is a multiple of the vector size.
        static_assert(Dh_MAX % K_VEC_SIZE == 0);
        // The type of queries and keys for the math in the Q*K^T product.
        using K_vec_k = typename K_vec_k_<T, K_VEC_SIZE>::Type;
        // Only used when key cache is quantized to 4 or 8 bits.
        constexpr int K_VEC_M_SIZE = K_VEC_SIZE / (ENABLE_4BITS_CACHE ? 2 : 1);
        using K_vec_m = typename packed_type<Tcache, K_VEC_M_SIZE>::type;

        // Use alignment for safely casting the shared buffers as Qk_vec_k and K_vec_k.
        // Shared memory to store Q inputs.
        __shared__ __align__(const_max(sizeof(Qk_vec_k), sizeof(K_vec_k))) Tk q_smem[Dh_MAX];

        // Make sure the hidden dimension per head is a multiple of the number of threads per value.
        static_assert(Dh_MAX % THREADS_PER_VALUE == 0); // trivially satisfied since THREADS_PER_VALUE == Dh_MAX / p

        // The number of elements per vector.
        constexpr int V_VEC_SIZE = Dh_MAX / THREADS_PER_VALUE;
        // A vector of V elements for the current timestep.
        using V_vec_k = typename V_vec_k_<T, V_VEC_SIZE>::Type;
        // Only used when value cache is quantized to 4 or 8 bits.
        constexpr int V_VEC_M_SIZE = V_VEC_SIZE / (ENABLE_4BITS_CACHE ? 2 : 1);
        using V_vec_m = typename packed_type<Tcache, V_VEC_M_SIZE>::type;

        static_assert(V_VEC_SIZE == sizeof(V_vec_k) / sizeof(T));

        // This could be one of the reasons to have a separate kernel for cross attention
        // constexpr auto bias_smem_size = 1u;     // constexpr auto bias_smem_size = DO_CROSS_ATTENTION ? Dh_MAX : 1u;

        // __shared__ __align__(const_max(const_max(sizeof(Qk_vec_k), sizeof(K_vec_k)), sizeof(V_vec_k)))
        //     Tk bias_smem[bias_smem_size];

        // The number of elements per vector.
        constexpr unsigned QK_VEC_SIZE{sizeof(Qk_vec_m) / sizeof(T)};
        // Make sure the hidden size per head is a multiple of the vector size.
        static_assert(Dh_MAX % QK_VEC_SIZE == 0);
        // We will use block wide reduction if needed
        // The number of vectors per Dh_MAX.
        constexpr unsigned QK_VECS_PER_Dh_MAX{Dh_MAX / QK_VEC_SIZE};
        static_assert(THREADS_PER_BLOCK >= QK_VECS_PER_Dh_MAX);

        // The batch/beam idx
        const auto bi = blockIdx.y;
        // half *k_scale_quant_orig_ptr = params.k_scale_quant_orig[bi];
        // half *v_scale_quant_orig_ptr = params.v_scale_quant_orig[bi];
        if (params.finished != nullptr && params.finished[bi])
        {
            return;
        }
        // The head.
        const unsigned hi{blockIdx.x};
        // The head index of keys and values adjusted for MQA/GQA.
        const int qhead_per_kv{params.num_heads / params.num_kv_heads};
        const unsigned hi_kv{hi / qhead_per_kv};
        // The number of heads.
        const auto num_heads = static_cast<unsigned>(params.num_heads);
        // The number of heads for keys and values adjusted for MQA/GQA.
        const auto num_heads_kv = static_cast<unsigned>(params.num_kv_heads);

        // The thread in the block.
        const unsigned tidx{threadIdx.x};

        // The column tile along L dimension on K^T -- noted as T_c in flash-attention paper
        const unsigned c_tile{0}; // const unsigned c_tile{MULTI_BLOCK_FLAG ? blockIdx.z : 0};

        // Indicate if we need to compute the K/V cache element (add KV bias, IA3, RoPE, etc.) and update the cache.
        // For Self-Attention, it's always required.
        // For Cross-Attention, as everything is pre-computed,
        // in the context phase of the encoder, it's not needed in that kernel.
        // Therefore, handle_kv is !DO_CROSS_ATTENTION and irrelevant of timestep.
        const bool handle_kv = true;  // const bool handle_kv{!DO_CROSS_ATTENTION};

        // While doing the product Q*K^T for the different keys we track the max.
        float qk_max = -FLT_MAX;

        float qk = 0.0F;

        // Compute relative attention bias on the fly, with relative attention table [head_num/TP, num_buckets] passed in.
        // num_buckets passed as params.relative_attention_bias_stride, max_distance passed as params.max_distance
        bool implicit_rel_attn_bias = params.max_distance != 0;
        int relative_attention_bias_stride = params.relative_attention_bias_stride; // num_buckets might be modified below, save it beforehand
        int max_distance = params.max_distance;

        // The actual sequence length excluding the paddings.
        // minus 1 because it includes the current timestep while tlength denotes the kv cache length.
        // const int tlength = DO_CROSS_ATTENTION
        //                         ? params.memory_length_per_sample[bi] - 1
        //                         : (params.length_per_sample ? (params.length_per_sample[bi] - 1) : static_cast<int>(timestep));
        const int tlength = (params.length_per_sample ? (params.length_per_sample[bi] - 1) : static_cast<int>(timestep));
        // The context length for beam searching optimization (all points to beam 0).
        const int input_length = params.input_lengths[bi];

        // The offset in the Q and K buffer also accounts for the batch.
        const auto qk_vec_idx = tidx * QK_VEC_SIZE;
        const auto is_valid_qk_vec = qk_vec_idx < Dh;

        // const bool load_qkv_quant = params.qkv_scale_quant_orig != nullptr;
        const bool write_attention_quant = params.attention_out_scale_orig_quant != nullptr;

        // Quant/Dequant scales for 8bits kv cache.
        using T_scale = typename kv_cache_scale_type_t<T, Tcache>::Type;
        T_scale kv_scale_quant_orig[2];
        T_scale kv_scale_orig_quant[2];

        constexpr int MAX_TIMESTEP_SCALES = SMEM_PRELOAD ? 2048 : 1;
        __shared__ half k_scales_history_smem[MAX_TIMESTEP_SCALES], k_zeros_history_smem[MAX_TIMESTEP_SCALES], v_scales_history_smem[MAX_TIMESTEP_SCALES], v_zeros_history_smem[MAX_TIMESTEP_SCALES];
        
        if constexpr (SMEM_PRELOAD)
        {
            int cur_timestep_idx = threadIdx.x * 8;
            Tcache *k_cache_ptr = reinterpret_cast<Tcache *>(kvCacheBuffer.getKBlockPtr(bi, cur_timestep_idx));
            half *k_scale_quant_orig_local_ptr = reinterpret_cast<half *>(k_cache_ptr + kvCacheBuffer.mBytesPerSeq);
            half *k_zeros_local_ptr = k_scale_quant_orig_local_ptr + kvCacheBuffer.mTokensPerBlock * num_heads_kv;
            Tcache *v_cache_ptr = reinterpret_cast<Tcache *>(kvCacheBuffer.getVBlockPtr(bi, cur_timestep_idx));
            half *v_scale_quant_orig_local_ptr = reinterpret_cast<half *>(v_cache_ptr + kvCacheBuffer.mBytesPerSeq);
            half *v_zeros_local_ptr = v_scale_quant_orig_local_ptr + kvCacheBuffer.mTokensPerBlock * num_heads_kv;
            // assume kscales stored as num_heads * num_tokens_per_block
            int k_scale_quant_orig_local_index = hi_kv * kvCacheBuffer.mTokensPerBlock + kvCacheBuffer.getLocalIdx(cur_timestep_idx);
            // if (cur_timestep_idx < tlength)
            // {
            //     *reinterpret_cast<uint4*>(k_scales_history_smem + cur_timestep_idx) = *(uint4*)(k_scale_quant_orig_local_ptr + k_scale_quant_orig_local_index);
            //     *reinterpret_cast<uint4*>(k_zeros_history_smem + cur_timestep_idx) = *(uint4*)(k_zeros_local_ptr + k_scale_quant_orig_local_index);
            //     *reinterpret_cast<uint4*>(v_scales_history_smem + cur_timestep_idx) = *(uint4*)(v_scale_quant_orig_local_ptr + k_scale_quant_orig_local_index);
            //     *reinterpret_cast<uint4*>(v_zeros_history_smem + cur_timestep_idx) = *(uint4*)(v_zeros_local_ptr + k_scale_quant_orig_local_index);
            // }
            // else
            // {
            //     *reinterpret_cast<uint4*>(k_scales_history_smem + cur_timestep_idx) = make_uint4(0, 0, 0, 0);
            //     *reinterpret_cast<uint4*>(k_zeros_history_smem + cur_timestep_idx) = make_uint4(0, 0, 0, 0);
            //     *reinterpret_cast<uint4*>(v_scales_history_smem + cur_timestep_idx) = make_uint4(0, 0, 0, 0);
            //     *reinterpret_cast<uint4*>(v_zeros_history_smem + cur_timestep_idx) = make_uint4(0, 0, 0, 0);

            // }
            
            bool ld_scale_zero_pred = cur_timestep_idx < tlength;
            if (ld_scale_zero_pred)
            {
                cp_async_launch(k_scales_history_smem + cur_timestep_idx, (uint4*)(k_scale_quant_orig_local_ptr + k_scale_quant_orig_local_index), ld_scale_zero_pred);
                cp_async_launch(k_zeros_history_smem + cur_timestep_idx, (uint4*)(k_zeros_local_ptr + k_scale_quant_orig_local_index), ld_scale_zero_pred);
                cp_async_launch(v_scales_history_smem + cur_timestep_idx, (uint4*)(v_scale_quant_orig_local_ptr + k_scale_quant_orig_local_index), ld_scale_zero_pred);
                cp_async_launch(v_zeros_history_smem + cur_timestep_idx, (uint4*)(v_zeros_local_ptr + k_scale_quant_orig_local_index), ld_scale_zero_pred);
                __pipeline_commit();
            }

            // __pipeline_wait_prior(0);
            // __syncthreads();
        }


        // #pragma unroll
        //         for (int i = 0; i < 2; i++)
        //         {
        //             convert_from_float(&kv_scale_quant_orig[i], (ENABLE_8BITS_CACHE ? params.kv_scale_quant_orig[i] : 1.0f));
        //         }
        // #pragma unroll
        //         for (int i = 0; i < 2; i++)
        //         {
        //             convert_from_float(&kv_scale_orig_quant[i], (ENABLE_8BITS_CACHE ? params.kv_scale_orig_quant[i] : 1.0f));
        //         }

        // Up to QK_VECS_PER_Dh_MAX threads load Q and K + the bias values for the current timestep.
        // Trigger the loads from the Q and K buffers.
        Qk_vec_k q, k; //, q_bias, k_bias;
        zero(q);
        zero(k);
        // zero(q_bias);
        // zero(k_bias);
        float rotary_embedding_base = params.rotary_embedding_base;
        float rotary_embedding_scale = params.rotary_embedding_scale;
        if (is_valid_qk_vec)
        {
            update_rotary_base_n_scale(rotary_embedding_base, rotary_embedding_scale,
                                       params.rotary_embedding_scale_type, params.rotary_embedding_dim, params.rotary_embedding_max_positions,
                                       tlength);
            // Query
            // The stride between tokens. We may be able to always use params.stride.
            uint32_t q_stride = params.stride ? static_cast<uint32_t>(params.stride) : (num_heads * Dh);
            // The offset.
            const auto q_offset = flat_index_strided3(bi, hi, qk_vec_idx, q_stride, Dh);

            // Note (shang): Load the current qk here. Not the quantized kv cache.
            {
                // Removed a branch for load_qkv_quant (current step qkv)
                q = vec_conversion<Qk_vec_k, Qk_vec_m>(*reinterpret_cast<const Qk_vec_m *>(&params.q[q_offset]));
            }
            {
                // Removed DO_CROSS_ATTENTION branch
                // Key
                // The stride between tokens. We may be able to always use params.stride.
                uint32_t k_stride = params.stride ? static_cast<uint32_t>(params.stride) : (num_heads_kv * Dh);
                // The offset.
                const auto k_offset = flat_index_strided3(bi, hi_kv, qk_vec_idx, k_stride, Dh);
                {
                    // Removed a branch for load_qkv_quant (current step qkv)
                    k = vec_conversion<Qk_vec_k, Qk_vec_m>(*reinterpret_cast<const Qk_vec_m *>(&params.k[k_offset]));
                }
            }

            // if (params.q_bias != nullptr)
            // {
            //     const auto q_bias_offset = flat_index2(hi, qk_vec_idx, Dh);
            //     q_bias = vec_conversion<Qk_vec_k, Qk_vec_m>(*reinterpret_cast<const Qk_vec_m *>(&params.q_bias[q_bias_offset]));
            // }
            // if (handle_kv && params.k_bias != nullptr)
            // {
            //     const auto k_bias_offset = flat_index2(hi_kv, qk_vec_idx, Dh);
            //     k_bias = vec_conversion<Qk_vec_k, Qk_vec_m>(*reinterpret_cast<const Qk_vec_m *>(&params.k_bias[k_bias_offset]));
            // }
        }

        const auto v_idx = chunk_index<T, V_vec_k, THREADS_PER_VALUE>(tidx);
        // The value computed by this thread.
        const auto vo = v_idx.x;
        // The hidden dimensions computed by this particular thread.
        const auto vi = v_idx.y;

        uint32_t v_stride = params.stride ? static_cast<uint32_t>(params.stride) : (num_heads_kv * Dh);
        const auto v_offset = flat_index_strided3(bi, hi_kv, vi, v_stride, Dh);

        V_vec_k v; // , v_bias;
        zero(v);
        // zero(v_bias);
        {
            // Removed a branch for load_qkv_quant (current step qkv)
            v = *reinterpret_cast<const V_vec_k *>(&params.v[v_offset]);
        }
        // if (handle_kv && params.v_bias != nullptr)
        // {
        //     const auto v_bias_offset = flat_index2(hi_kv, qk_vec_idx, Dh);
        //     v_bias = *reinterpret_cast<const V_vec_k *>(&params.v_bias[v_bias_offset]);
        // }
        // v = add(v, v_bias);
        half *v_scale_block_ptr = reinterpret_cast<half *>(kvCacheBuffer.getVBlockPtr(bi, tlength) + kvCacheBuffer.mBytesPerSeq);
        half *v_scale_cur_ptr = v_scale_block_ptr + hi_kv * kvCacheBuffer.mTokensPerBlock + kvCacheBuffer.getLocalIdx(tlength);

        float v_max, v_min, v_scale_orig_quant, v_zeros;
        //__shared__ half v_scales_smem[1], v_zeros_smem[1];
        __shared__ half2 v_sz_smem[1];
        if constexpr (ENABLE_ZEROS)
        {
            half *v_zero_cur_ptr = v_scale_cur_ptr + num_heads_kv * kvCacheBuffer.mTokensPerBlock;
            v_max = vec_max_no_abs<V_vec_k>(v);
            v_min = vec_min_no_abs<V_vec_k>(v);
            assert(QK_VECS_PER_Dh_MAX <= WARP_SIZE);
#pragma unroll
            for (int mask = QK_VECS_PER_Dh_MAX / 2; mask >= 1; mask /= 2)
            {
                v_max = fmaxf(v_max, __shfl_xor_sync(shfl_mask(QK_VECS_PER_Dh_MAX), v_max, mask));
                v_min = fminf(v_min, __shfl_xor_sync(shfl_mask(QK_VECS_PER_Dh_MAX), v_min, mask));
            }
            if constexpr (ENABLE_4BITS_CACHE)
            {
                if (tidx == 0)
                {
                    *v_sz_smem = make_half2(__float2half_rn((v_max - v_min) / 15), __float2half_rn(-15.0f * v_min / (v_max - v_min)));
                }
            }
            else
            {
                if (tidx == 0)
                {
                    *v_sz_smem = make_half2(__float2half_rn((v_max - v_min) / 255), __float2half_rn(-255.0f * v_min / (v_max - v_min)));
                }
            }
            __syncthreads();
            v_scale_orig_quant = 1.0f / __half2float((*v_sz_smem).x);
            v_zeros = __half2float((*v_sz_smem).y);
            *v_scale_cur_ptr = (*v_sz_smem).x;
            *v_zero_cur_ptr = (*v_sz_smem).y;
        }
        else
        {

            v_max = vec_max<V_vec_k>(v);
            // tree reduction for final results (within a warp)
            assert(QK_VECS_PER_Dh_MAX <= WARP_SIZE);
#pragma unroll
            for (int mask = QK_VECS_PER_Dh_MAX / 2; mask >= 1; mask /= 2)
            {
                v_max = fmaxf(v_max, __shfl_xor_sync(shfl_mask(QK_VECS_PER_Dh_MAX), v_max, mask));
            }
            // wb to DRAM
            if constexpr (ENABLE_4BITS_CACHE)
            {
                if (tidx == 0)
                {
                    // params.v_scale_quant_orig[(bi * max_seq_len + tlength) * num_heads_kv + hi_kv] = __float2half_rn(v_max / 127);
                    *v_scale_cur_ptr = __float2half_rn(v_max / 7);
                }
            }
            else
            {
                if (tidx == 0)
                {
                    // params.v_scale_quant_orig[(bi * max_seq_len + tlength) * num_heads_kv + hi_kv] = __float2half_rn(v_max / 127);
                    *v_scale_cur_ptr = __float2half_rn(v_max / 127);
                }
            }
            __syncthreads();
            // float v_scale_orig_quant = 1.0f / __half2float(params.v_scale_quant_orig[(bi * max_seq_len + tlength) * num_heads_kv + hi_kv]);
            v_scale_orig_quant = 1.0f / __half2float(*v_scale_cur_ptr);
        }
        // Computes the Q/K values with bias.
        // q = add(q, q_bias);
        // if (handle_kv)
        // {
        //     k = add(k, k_bias);
        // }

        // const bool do_ia3 = handle_kv && params.ia3_tasks != nullptr;
        const auto beam_width = static_cast<unsigned>(params.beam_width);
        // const auto ia3_ti_hi = do_ia3
        //                            ? flat_index2(static_cast<unsigned>(params.ia3_tasks[bi / beam_width]), hi, num_heads)
        //                            : 0;

        // if (do_ia3 && is_valid_qk_vec)
        // {
        //     k = mul<Qk_vec_k, Qk_vec_k, Qk_vec_k>(k,
        //                                           vec_conversion<Qk_vec_k, Qk_vec_m>(*reinterpret_cast<const Qk_vec_m *>(
        //                                               &params.ia3_key_weights[flat_index2(ia3_ti_hi, qk_vec_idx, Dh)])));
        // }

        // Note we have no paddings in KV cache now.
        // switch (params.position_embedding_type)
        // {
        // case PositionEmbeddingType::kLEARNED_ABSOLUTE:
        // case PositionEmbeddingType::kRELATIVE:
        // case PositionEmbeddingType::kALIBI:
        // case PositionEmbeddingType::kALIBI_WITH_SCALE:
        //     break;
        // case PositionEmbeddingType::kROPE_GPTJ:
        // {
        //     if (handle_kv)
        //     {
        //         apply_rotary_embedding(q, k, tidx, params.rotary_embedding_dim, params.rotary_embedding_base,
        //                                params.rotary_embedding_scale, tlength);
        //     }
        //     else
        //     {
        //         apply_rotary_embedding(q, tidx, params.rotary_embedding_dim, params.rotary_embedding_base,
        //                                params.rotary_embedding_scale, tlength);
        //     }
        //     break;
        // }
        // case PositionEmbeddingType::kROPE_GPT_NEOX:
        {
            const bool do_rotary = is_valid_qk_vec && QK_VEC_SIZE * tidx < params.rotary_embedding_dim;

            T *q_smem_ = reinterpret_cast<T *>(smem_);
            T *k_smem = q_smem_ + params.rotary_embedding_dim;

            const int half_rotary_dim = params.rotary_embedding_dim / 2;
            const int half_idx = qk_vec_idx / half_rotary_dim;
            const int intra_half_idx = qk_vec_idx % half_rotary_dim;
            const int smem_pitch = half_rotary_dim; // TODO: adjust for bank conflicts

            assert(half_rotary_dim % QK_VEC_SIZE == 0);

            if (do_rotary)
            {
                *reinterpret_cast<Qk_vec_k *>(q_smem_ + half_idx * smem_pitch + intra_half_idx) = q;
                if (handle_kv)
                {
                    *reinterpret_cast<Qk_vec_k *>(k_smem + half_idx * smem_pitch + intra_half_idx) = k;
                }
            }

            __syncthreads();

            const int transpose_idx = half_idx * (half_rotary_dim / 2) + intra_half_idx / 2;
            constexpr int tidx_factor = (QK_VEC_SIZE > 1) ? QK_VEC_SIZE / 2 : 1;
            if (do_rotary)
            {
                vec_from_smem_transpose(q, q_smem_, transpose_idx, smem_pitch);
                if (handle_kv)
                {
                    vec_from_smem_transpose(k, k_smem, transpose_idx, smem_pitch);

                    apply_rotary_embedding(q, k, transpose_idx / tidx_factor, params.rotary_embedding_dim,
                                           rotary_embedding_base, rotary_embedding_scale, tlength);

                    write_smem_transpose(k, k_smem, transpose_idx, smem_pitch);
                }
                else
                {
                    apply_rotary_embedding(q, transpose_idx / tidx_factor, params.rotary_embedding_dim,
                                           rotary_embedding_base, rotary_embedding_scale, tlength);
                }
                write_smem_transpose(q, q_smem_, transpose_idx, smem_pitch);
            }

            __syncthreads();

            if (do_rotary)
            {
                q = *reinterpret_cast<Qk_vec_k *>(q_smem_ + half_idx * smem_pitch + intra_half_idx);
                if (handle_kv)
                {
                    k = *reinterpret_cast<Qk_vec_k *>(k_smem + half_idx * smem_pitch + intra_half_idx);
                }
            }

            __syncthreads();
            // break;
        }
        // }

        // Base pointer to k cache block for beam's batch
        half *k_scale_block_ptr = reinterpret_cast<half *>(kvCacheBuffer.getKBlockPtr(bi, tlength) + kvCacheBuffer.mBytesPerSeq);
        half *k_scale_cur_ptr = k_scale_block_ptr + hi_kv * kvCacheBuffer.mTokensPerBlock + kvCacheBuffer.getLocalIdx(tlength);        // TODO (Haotian): calculate k_scale_orig_quant, per-head max for k
        // reduction within single thread for k
        float k_max, k_min, k_scale_orig_quant, k_zeros;
        // __shared__ half k_scales_smem[1], k_zeros_smem[1];
        __shared__ half2 k_sz_smem[1];
        if constexpr (ENABLE_ZEROS)
        {
            half *k_zero_cur_ptr = k_scale_cur_ptr + num_heads_kv * kvCacheBuffer.mTokensPerBlock;
            k_max = vec_max_no_abs<Qk_vec_m>(k);
            k_min = vec_min_no_abs<Qk_vec_m>(k);
            assert(QK_VECS_PER_Dh_MAX <= WARP_SIZE);
#pragma unroll
            for (int mask = QK_VECS_PER_Dh_MAX / 2; mask >= 1; mask /= 2)
            {
                k_max = fmaxf(k_max, __shfl_xor_sync(shfl_mask(QK_VECS_PER_Dh_MAX), k_max, mask));
                k_min = fminf(k_min, __shfl_xor_sync(shfl_mask(QK_VECS_PER_Dh_MAX), k_min, mask));
            }
            if constexpr (ENABLE_4BITS_CACHE)
            {
                if (tidx == 0)
                {
                    *k_sz_smem = make_half2(__float2half_rn((k_max - k_min) / 15), __float2half_rn(-15.0f * k_min / (k_max - k_min)));
                }
            }
            else
            {
                if (tidx == 0)
                {
                    *k_sz_smem = make_half2(__float2half_rn((k_max - k_min) / 255), __float2half_rn(-255.0f * k_min / (k_max - k_min)));
                }
            }
            __syncthreads();
            k_scale_orig_quant = 1.0f / __half2float((*k_sz_smem).x);
            k_zeros = __half2float((*k_sz_smem).y);
            *k_scale_cur_ptr = (*k_sz_smem).x;
            *k_zero_cur_ptr = (*k_sz_smem).y;
        }
        else
        {
            k_max = vec_max<Qk_vec_m>(k);
            // tree reduction for final results (within a warp)
            assert(QK_VECS_PER_Dh_MAX <= WARP_SIZE);
#pragma unroll
            for (int mask = QK_VECS_PER_Dh_MAX / 2; mask >= 1; mask /= 2)
            {
                k_max = fmaxf(k_max, __shfl_xor_sync(shfl_mask(QK_VECS_PER_Dh_MAX), k_max, mask));
            }
            if constexpr (ENABLE_4BITS_CACHE)
            {
                // wb to DRAM
                if (tidx == 0)
                {
                    // params.k_scale_quant_orig[(bi * max_seq_len + tlength) * num_heads_kv + hi_kv] = __float2half_rn(k_max / 127);
                    *k_scale_cur_ptr = __float2half_rn(k_max / 7);
                }
            }
            else
            {
                // wb to DRAM
                if (tidx == 0)
                {
                    // params.k_scale_quant_orig[(bi * max_seq_len + tlength) * num_heads_kv + hi_kv] = __float2half_rn(k_max / 127);
                    *k_scale_cur_ptr = __float2half_rn(k_max / 127);
                }
            }
            __syncthreads();
            // float k_scale_orig_quant = 1.0f / __half2float(params.k_scale_quant_orig[(bi * max_seq_len + tlength) * num_heads_kv + hi_kv]);
            k_scale_orig_quant = 1.0f / __half2float(*k_scale_cur_ptr);
        }
        // For the same reason as handle_kv, no compute needed in Cross-Attention's 1st step
        if (qk_vec_idx < Dh_MAX)
        {

            // Store the Q values to shared memory.
            // Set padded Dh to 0 for the correctness of QK (when Dh != Dh_Max).
            Qk_vec_k zero_q;
            zero(zero_q);

            *reinterpret_cast<Qk_vec_k *>(&q_smem[qk_vec_idx]) = is_valid_qk_vec ? q : zero_q;

            // Write the K values to the global memory cache.
            //
            // NOTE: The stores are uncoalesced as we have multiple chunks of 16B spread across the memory
            // system. We designed it this way as it allows much better memory loads (and there are many
            // more loads) + the stores are really "write and forget" since we won't need the ack before
            // the end of the kernel. There's plenty of time for the transactions to complete.

            // For MQA/GQA mode, write only with the first Q head of each group per KV head.
            if (handle_kv && hi == (hi_kv * qhead_per_kv) && (IS_Dh_MAX || is_valid_qk_vec))
            {
                // Trigger the stores to global memory.
                const auto k_idx = QK_VEC_SIZE * tidx;

                const int inBlockIdx = kvCacheBuffer.getKVLocalIdx(tlength, hi_kv, Dh / (ENABLE_4BITS_CACHE ? 2 : 1), k_idx / (ENABLE_4BITS_CACHE ? 2 : 1));
                // The base pointer for the value in the cache buffer.
                Tcache *k_cache = reinterpret_cast<Tcache *>(kvCacheBuffer.getKBlockPtr(bi, tlength));
                if constexpr (ENABLE_ZEROS)
                {
                    if constexpr (ENABLE_4BITS_CACHE)
                    {
                        store_4bits_kv_cache_vec(reinterpret_cast<Tcache *>(k_cache), k, inBlockIdx, k_scale_orig_quant, k_zeros);
                    }
                    else if constexpr (ENABLE_8BITS_CACHE)
                    {
                        store_8bits_kv_cache_vec(reinterpret_cast<Tcache *>(k_cache), k, inBlockIdx, k_scale_orig_quant, k_zeros);
                    }
                    else
                    {
                        *reinterpret_cast<Qk_vec_m *>(&k_cache[inBlockIdx]) = vec_conversion<Qk_vec_m, Qk_vec_k>(k);
                    }
                }
                else
                {
                    if constexpr (ENABLE_4BITS_CACHE)
                    {
                        store_4bits_kv_cache_vec(reinterpret_cast<Tcache *>(k_cache), k, inBlockIdx, k_scale_orig_quant);
                    }
                    else if constexpr (ENABLE_8BITS_CACHE)
                    {
                        store_8bits_kv_cache_vec(reinterpret_cast<Tcache *>(k_cache), k, inBlockIdx, k_scale_orig_quant);
                    }
                    else
                    {
                        *reinterpret_cast<Qk_vec_m *>(&k_cache[inBlockIdx]) = vec_conversion<Qk_vec_m, Qk_vec_k>(k);
                    }
                }
            }

            // Compute \sum_i Q[i] * K^T[i] for the current timestep.
#ifdef MMHA_USE_FP32_ACUM_FOR_FMA
            using Qk_vec_acum = typename Qk_vec_acum_fp32_<Qk_vec_k>::Type;
#else
            using Qk_vec_acum = Qk_vec_k;
#endif
            qk = dot<Qk_vec_acum, Qk_vec_k>(q, k);
            if constexpr (QK_VECS_PER_Dh_MAX <= WARP_SIZE)
            {
#pragma unroll
                for (int mask = QK_VECS_PER_Dh_MAX / 2; mask >= 1; mask /= 2)
                {
                    qk += __shfl_xor_sync(shfl_mask(QK_VECS_PER_Dh_MAX), qk, mask);
                }
            }
        }

        if constexpr (QK_VECS_PER_Dh_MAX > WARP_SIZE)
        {
            constexpr int WARPS_PER_RED = (QK_VECS_PER_Dh_MAX + WARP_SIZE - 1) / WARP_SIZE;
            qk = block_sum<WARPS_PER_RED>(&red_smem[WARPS_PER_RED], qk);
        }

        // Store that value in shared memory. Keep the Q*K^T value in register for softmax.
        if (tidx == 0)
        {
            // Normalize qk.
            qk *= params.inv_sqrt_dh;
            // if (params.relative_attention_bias != nullptr)
            // {
            //     if (implicit_rel_attn_bias)
            //     {
            //         // Here i == j == tlength, so relative_position = 0 --> relative_buckets = 0.
            //         T rel_attn_bias = params.relative_attention_bias[hi * relative_attention_bias_stride + 0];
            //         qk = add(qk, rel_attn_bias);
            //     }
            //     else
            //     {
            //         qk = add(qk,
            //                  params.relative_attention_bias[hi * params.relative_attention_bias_stride * params.relative_attention_bias_stride + tlength * params.relative_attention_bias_stride + tlength]);
            //     }
            // }
            // We don't need to apply the linear position bias here since qi - ki = 0 yields the position bias 0.

            qk_max = qk;
            // qk_smem[params.timestep] = qk;
            {
                // removed branch for MULTI_BLOCK_FLAG
                qk_smem[tlength] = qk;
            }
        }

        // Make sure the data is in shared memory.
        __syncthreads();

        constexpr unsigned K_ELTS_PER_CHUNK{THREADS_PER_KEY * K_VEC_SIZE};

        // The positions of the cache buffer (for this B * H) and the vector within that chunk associated with this
        // thread.
        const auto k_idx = chunk_index<T, K_vec_k, THREADS_PER_KEY>(tidx);

        // The number of vectors per thread.
        constexpr unsigned K_VECS_PER_THREAD{Dh_MAX / K_ELTS_PER_CHUNK};
        static_assert(Dh_MAX == K_ELTS_PER_CHUNK * K_VECS_PER_THREAD);

        // Load the Q values from shared memory. The values are reused during the loop on K.
        K_vec_k q_vec[K_VECS_PER_THREAD];
if constexpr (ENABLE_4BITS_CACHE && ENABLE_ZEROS)
        {
    #pragma unroll
            for (unsigned ii = 0; ii < K_VECS_PER_THREAD; ++ii)
            {
                q_vec[ii] = reorder_8xfp16(*reinterpret_cast<const K_vec_k *>(
                    &q_smem[flat_index2(ii, k_idx.y, K_ELTS_PER_CHUNK)]));
            }
        }
        else
        {
            #pragma unroll
            for (unsigned ii = 0; ii < K_VECS_PER_THREAD; ++ii)
            {
                q_vec[ii] = *reinterpret_cast<const K_vec_k *>(
                    &q_smem[flat_index2(ii, k_idx.y, K_ELTS_PER_CHUNK)]);
            }
        }
        // The number of timesteps loaded per iteration, i.e., (THREADS_PER_BLOCK * THREADS_PER_BLOCK) / 256 <= 256
        constexpr unsigned K_PER_ITER{THREADS_PER_BLOCK / THREADS_PER_KEY};
        // The number of keys per warp.
        constexpr unsigned K_PER_WARP{WARP_SIZE / THREADS_PER_KEY};
        // The number of unrolled keys per warp.
        constexpr unsigned UNROLLED_K_PER_WARP = K_PER_WARP * K_LOOP_UNROLL;
        // The number of unrolled keys per ieration.
        constexpr unsigned UNROLLED_K_PER_ITER = K_PER_ITER * K_LOOP_UNROLL;

        // Base pointer for the row of pointers to k cache blocks
        void **k_cache_base_row_ptr = reinterpret_cast<void **>(kvCacheBuffer.getRowPtr(KVIdxType::K_IDX, bi));

        const auto timesteps_per_block = static_cast<unsigned>(params.timesteps_per_block);

        // Pick a number of keys to make sure all the threads of a warp enter (due to shfl_sync).
        // Take all previous cache as context when we have no beam searching in order to batch as many LDGs as possible.
        const int context_length = tlength;  // const int context_length = HAS_BEAMS ? input_length : tlength;
        // const auto context_ti_end = MULTI_BLOCK_FLAG
        //                                 ? divUp(timesteps_per_block, UNROLLED_K_PER_WARP) * UNROLLED_K_PER_WARP
        //                                 : divUp(static_cast<unsigned>(context_length), UNROLLED_K_PER_WARP) * UNROLLED_K_PER_WARP;
        const auto context_ti_end = divUp(static_cast<unsigned>(context_length), UNROLLED_K_PER_WARP) * UNROLLED_K_PER_WARP;

        // The generation ti_end.
        // const auto generation_ti_end = MULTI_BLOCK_FLAG ? divUp(timesteps_per_block, K_PER_WARP) * K_PER_WARP
        //                                                 : divUp(static_cast<unsigned>(tlength), K_PER_WARP) * K_PER_WARP;
        const auto generation_ti_end = divUp(static_cast<unsigned>(tlength), K_PER_WARP) * K_PER_WARP;

        // Iterate over the keys/timesteps to compute the various (Q*K^T)_{ti} values.
        const auto bi_seq_len_offset = static_cast<std::size_t>(bi) * max_seq_len;

        const auto c_tile_times_timesteps_per_block = c_tile * timesteps_per_block; // 0 if !MULTI_BLOCK_FLAG

        ////////////////////////////////////////////////////////////////////////////////////////////////
        // Key cache loops for dot(Q, K).

        // Handle only context key cache with beam searching.
        // Handle both context and generation key cache without beam searching.
        // Explict batching of LDGs (by K_LOOP_UNROLL) as it doesn't depend on indirection tables.

        if constexpr (ENABLE_ZEROS)
        {
            // Now we will use kscales, kzeros, etc. so we need pipeline wait prior
            if constexpr (SMEM_PRELOAD)
            {
                __pipeline_wait_prior(0);
            }
            for (int ti = k_idx.x; ti < context_ti_end; ti += UNROLLED_K_PER_ITER)
            {
                // const int time_now = MULTI_BLOCK_FLAG ? ti + c_tile_times_timesteps_per_block : ti;
                const int time_now = ti;

                // The keys loaded from the key cache.
                K_vec_m k_vec_cache[K_LOOP_UNROLL][K_VECS_PER_THREAD];
                float k_scale_quant_orig_local[K_LOOP_UNROLL];
                float k_zeros_local[K_LOOP_UNROLL];

#pragma unroll
                for (int k_loop = 0; k_loop < K_LOOP_UNROLL; ++k_loop)
                {
                    // Haotian: we probably do not need this because each page also contains slots for OOB tokens
                    // const int valid_time_now = min(time_now + k_loop * K_PER_ITER, context_length - 1);
                    const int valid_time_now = time_now + k_loop * K_PER_ITER;

                    // const int seqIdx = bi / beam_width * beam_width;
                    const int seqIdx = bi;
                    // Base pointer to k cache block fo r beam's batch
                    Tcache *k_cache_batch = reinterpret_cast<Tcache *>(kvCacheBuffer.getKBlockPtr(seqIdx, valid_time_now));
                    half *k_scale_quant_orig_local_ptr = reinterpret_cast<half *>(k_cache_batch + kvCacheBuffer.mBytesPerSeq);
                    half *k_zeros_local_ptr = k_scale_quant_orig_local_ptr + kvCacheBuffer.mTokensPerBlock * num_heads_kv;
                    int k_scale_quant_orig_local_index = hi_kv * kvCacheBuffer.mTokensPerBlock + kvCacheBuffer.getLocalIdx(valid_time_now);
                    // k_scale_quant_orig_local[k_loop] = __half2float(params.k_scale_quant_orig[(seqIdx * max_seq_len + valid_time_now) * num_heads_kv + hi_kv]);
                    if constexpr (SMEM_PRELOAD)
                    {
                        k_scale_quant_orig_local[k_loop] = k_scales_history_smem[valid_time_now];//__half2float(k_scale_quant_orig_local_ptr[k_scale_quant_orig_local_index]);
                        k_zeros_local[k_loop] = k_zeros_history_smem[valid_time_now];//__half2float(k_zeros_local_ptr[k_scale_quant_orig_local_index]);
                    }
                    else
                    {
                        k_scale_quant_orig_local[k_loop] = __half2float(k_scale_quant_orig_local_ptr[k_scale_quant_orig_local_index]);
                        k_zeros_local[k_loop] = __half2float(k_zeros_local_ptr[k_scale_quant_orig_local_index]);
                    }
#pragma unroll
                    for (int k_vec_i = 0; k_vec_i < K_VECS_PER_THREAD; ++k_vec_i)
                    {
                        // Make sure we read data within the bound.
                        // Dh OOB values will be handled by zero_q.
                        // Seq OOB values will be masked out when storing back to smem.
                        auto const jj = min(k_idx.y + k_vec_i * K_ELTS_PER_CHUNK, Dh - K_VEC_SIZE);

                        int inBlockIdx = kvCacheBuffer.getKVLocalIdx(valid_time_now, hi_kv, Dh / (ENABLE_4BITS_CACHE ? 2 : 1), jj / (ENABLE_4BITS_CACHE ? 2 : 1));
                        k_vec_cache[k_loop][k_vec_i] = *reinterpret_cast<const K_vec_m *>(&k_cache_batch[inBlockIdx]);
                    }
                }

#pragma unroll
                for (int k_loop = 0; k_loop < K_LOOP_UNROLL; ++k_loop)
                {
                    const int local_time_now = time_now + k_loop * K_PER_ITER;
                    const int local_ti = ti + k_loop * K_PER_ITER;
                    float k_scale = k_scale_quant_orig_local[k_loop];
                    float k_zero = k_zeros_local[k_loop];

                    K_vec_k k_vec[K_VECS_PER_THREAD];
#pragma unroll
                    for (int k_vec_i = 0; k_vec_i < K_VECS_PER_THREAD; ++k_vec_i)
                    {
                        // TODO (shang): modify for 4bit kv
                        // we move quantization to here for better batching of inflight LDGs.
                        if constexpr (ENABLE_4BITS_CACHE)
                        {
                            convert_from_4bit_kv_cache<K_vec_m, K_vec_k, Tcache, T_scale>(
                                &k_vec[k_vec_i], k_vec_cache[k_loop][k_vec_i], k_scale, k_zero);
                        }
                        else if constexpr (ENABLE_8BITS_CACHE)
                        {
                            convert_from_8bit_kv_cache<K_vec_m, K_vec_k, Tcache, T_scale>(
                                &k_vec[k_vec_i], k_vec_cache[k_loop][k_vec_i], k_scale, k_zero);
                        }
                        else
                        {
                            // K_vek is same as K_vec_cache in this case.
                            k_vec[k_vec_i] = *reinterpret_cast<K_vec_k *>(&k_vec_cache[k_loop][k_vec_i]);
                        }
                    }

                    // Perform the dot product and normalize qk.
                    //
                    // WARNING: ALL THE THREADS OF A WARP MUST ENTER!!!
                    
                    // float qk_{Qk_dot<T, THREADS_PER_KEY>::dot(q_vec, k_vec) * params.inv_sqrt_dh};
                    assert (K_VECS_PER_THREAD == 1);
                    float qk_ = qk_hmma_dot_simple<THREADS_PER_KEY>(q_vec[0], k_vec[0]) * params.inv_sqrt_dh;

                    // For multi-block mode, we still need to make sure it will not be OOB.
                    // if (MULTI_BLOCK_FLAG && local_ti >= timesteps_per_block)
                    // {
                    //     continue;
                    // }

                    // Store the product to shared memory. There's one qk value per timestep. Update the max.
                    if (local_time_now < context_length && tidx % THREADS_PER_KEY == 0)
                    {
                        // if (params.relative_attention_bias != nullptr)
                        // {
                        //     if (implicit_rel_attn_bias)
                        //     {
                        //         // Compute bias value on the fly (See bert_preprocess_kernels.cu::buildRelativeAttentionBias)
                        //         int relative_buckets = 0;
                        //         int relative_position = local_time_now - tlength;
                        //         int num_buckets = relative_attention_bias_stride;
                        //         // Special logic in T5 relative attention, both encoder & decoder use this, because
                        //         // relative_attention_bias is pre-computed once and passed around.
                        //         num_buckets /= 2;
                        //         relative_buckets += relative_position > 0 ? num_buckets : 0;
                        //         relative_position = abs(relative_position);
                        //         int max_exact = num_buckets / 2;
                        //         bool is_small = relative_position < max_exact;
                        //         int relative_position_if_large = max_exact + (int)(logf(relative_position * 1.0f / max_exact) / logf((float)max_distance / max_exact) * (num_buckets - max_exact));
                        //         relative_position_if_large = min(relative_position_if_large, num_buckets - 1);
                        //         relative_buckets += is_small ? relative_position : relative_position_if_large;
                        //         T rel_attn_bias = params.relative_attention_bias[hi * relative_attention_bias_stride + relative_buckets];
                        //         qk_ = add(qk_, rel_attn_bias);
                        //     }
                        //     else
                        //     {
                        //         qk_ = add(qk_,
                        //                   params.relative_attention_bias[hi * params.relative_attention_bias_stride * params.relative_attention_bias_stride + tlength * params.relative_attention_bias_stride + local_time_now]);
                        //     }
                        // }
                        // if (params.linear_bias_slopes != nullptr)
                        // {
                        //     // Apply the linear position bias: (ki - qi) * slope[hi].
                        //     // The padding token locates between the input context and the generated tokens.
                        //     // We need to remove the number of padding tokens in the distance computation.
                        //     //   ti   : 0 1 2 3 4 5 6 7 8 9(tlength)
                        //     //   token: i i i i p p p o o o where i=input, p=pad, o=output.
                        //     // e.g. ti = 2, dist = (9 - 3) - 2 = 4.
                        //     float dist = local_time_now - tlength;

                        //     qk_ += mul<float, T, float>(params.linear_bias_slopes[hi], dist);
                        // }

                        // Calculate the max for softmax, and store qk back to smem.
                        // Don't need mask here as we remove paddings in kv cache.
                        qk_max = fmaxf(qk_max, qk_);
                        qk_smem[local_ti] = qk_;
                    }
                }
            }
        }
        else
        {

            for (int ti = k_idx.x; ti < context_ti_end; ti += UNROLLED_K_PER_ITER)
            {
                // const int time_now = MULTI_BLOCK_FLAG ? ti + c_tile_times_timesteps_per_block : ti;
                const int time_now = ti;

                // The keys loaded from the key cache.
                K_vec_m k_vec_cache[K_LOOP_UNROLL][K_VECS_PER_THREAD];
                float k_scale_quant_orig_local[K_LOOP_UNROLL];

#pragma unroll
                for (int k_loop = 0; k_loop < K_LOOP_UNROLL; ++k_loop)
                {
                    const int valid_time_now = min(time_now + k_loop * K_PER_ITER, context_length - 1);
                    // const int seqIdx = bi / beam_width * beam_width;
                    const int seqIdx = bi;
                    // Base pointer to k cache block fo r beam's batch
                    Tcache *k_cache_batch = reinterpret_cast<Tcache *>(kvCacheBuffer.getKBlockPtr(seqIdx, valid_time_now));
                    half *k_scale_quant_orig_local_ptr = reinterpret_cast<half *>(k_cache_batch + kvCacheBuffer.mBytesPerSeq);
                    int k_scale_quant_orig_local_index = kvCacheBuffer.getLocalIdx(valid_time_now) * num_heads_kv + hi_kv;
                    // k_scale_quant_orig_local[k_loop] = __half2float(params.k_scale_quant_orig[(seqIdx * max_seq_len + valid_time_now) * num_heads_kv + hi_kv]);
                    k_scale_quant_orig_local[k_loop] = __half2float(k_scale_quant_orig_local_ptr[k_scale_quant_orig_local_index]);
#pragma unroll
                    for (int k_vec_i = 0; k_vec_i < K_VECS_PER_THREAD; ++k_vec_i)
                    {
                        // Make sure we read data within the bound.
                        // Dh OOB values will be handled by zero_q.
                        // Seq OOB values will be masked out when storing back to smem.
                        auto const jj = min(k_idx.y + k_vec_i * K_ELTS_PER_CHUNK, Dh - K_VEC_SIZE);

                        int inBlockIdx = kvCacheBuffer.getKVLocalIdx(valid_time_now, hi_kv, Dh / (ENABLE_4BITS_CACHE ? 2 : 1), jj / (ENABLE_4BITS_CACHE ? 2 : 1));
                        k_vec_cache[k_loop][k_vec_i] = *reinterpret_cast<const K_vec_m *>(&k_cache_batch[inBlockIdx]);
                    }
                }

#pragma unroll
                for (int k_loop = 0; k_loop < K_LOOP_UNROLL; ++k_loop)
                {
                    const int local_time_now = time_now + k_loop * K_PER_ITER;
                    const int local_ti = ti + k_loop * K_PER_ITER;
                    float k_scale = k_scale_quant_orig_local[k_loop];

                    K_vec_k k_vec[K_VECS_PER_THREAD];
#pragma unroll
                    for (int k_vec_i = 0; k_vec_i < K_VECS_PER_THREAD; ++k_vec_i)
                    {
                        // TODO (shang): modify for 4bit kv
                        // we move quantization to here for better batching of inflight LDGs.
                        if constexpr (ENABLE_4BITS_CACHE)
                        {
                            convert_from_4bit_kv_cache<K_vec_m, K_vec_k, Tcache, T_scale>(
                                &k_vec[k_vec_i], k_vec_cache[k_loop][k_vec_i], k_scale);
                        }
                        else if constexpr (ENABLE_8BITS_CACHE)
                        {
                            convert_from_8bit_kv_cache<K_vec_m, K_vec_k, Tcache, T_scale>(
                                &k_vec[k_vec_i], k_vec_cache[k_loop][k_vec_i], k_scale);
                        }
                        else
                        {
                            // K_vek is same as K_vec_cache in this case.
                            k_vec[k_vec_i] = *reinterpret_cast<K_vec_k *>(&k_vec_cache[k_loop][k_vec_i]);
                        }
                    }

                    // Perform the dot product and normalize qk.
                    //
                    // WARNING: ALL THE THREADS OF A WARP MUST ENTER!!!
                    // float qk_{Qk_dot<T, THREADS_PER_KEY>::dot(q_vec, k_vec) * params.inv_sqrt_dh};
                    assert (K_VECS_PER_THREAD == 1);
                    float qk_ = qk_hmma_dot_simple<THREADS_PER_KEY>(q_vec[0], k_vec[0]) * params.inv_sqrt_dh;

                    // For multi-block mode, we still need to make sure it will not be OOB.
                    // if (MULTI_BLOCK_FLAG && local_ti >= timesteps_per_block)
                    // {
                    //     continue;
                    // }

                    // Store the product to shared memory. There's one qk value per timestep. Update the max.
                    if (local_time_now < context_length && tidx % THREADS_PER_KEY == 0)
                    {
                        // if (params.relative_attention_bias != nullptr)
                        // {
                        //     if (implicit_rel_attn_bias)
                        //     {
                        //         // Compute bias value on the fly (See bert_preprocess_kernels.cu::buildRelativeAttentionBias)
                        //         int relative_buckets = 0;
                        //         int relative_position = local_time_now - tlength;
                        //         int num_buckets = relative_attention_bias_stride;
                        //         // Special logic in T5 relative attention, both encoder & decoder use this, because
                        //         // relative_attention_bias is pre-computed once and passed around.
                        //         num_buckets /= 2;
                        //         relative_buckets += relative_position > 0 ? num_buckets : 0;
                        //         relative_position = abs(relative_position);
                        //         int max_exact = num_buckets / 2;
                        //         bool is_small = relative_position < max_exact;
                        //         int relative_position_if_large = max_exact + (int)(logf(relative_position * 1.0f / max_exact) / logf((float)max_distance / max_exact) * (num_buckets - max_exact));
                        //         relative_position_if_large = min(relative_position_if_large, num_buckets - 1);
                        //         relative_buckets += is_small ? relative_position : relative_position_if_large;
                        //         T rel_attn_bias = params.relative_attention_bias[hi * relative_attention_bias_stride + relative_buckets];
                        //         qk_ = add(qk_, rel_attn_bias);
                        //     }
                        //     else
                        //     {
                        //         qk_ = add(qk_,
                        //                   params.relative_attention_bias[hi * params.relative_attention_bias_stride * params.relative_attention_bias_stride + tlength * params.relative_attention_bias_stride + local_time_now]);
                        //     }
                        // }
                        // if (params.linear_bias_slopes != nullptr)
                        // {
                        //     // Apply the linear position bias: (ki - qi) * slope[hi].
                        //     // The padding token locates between the input context and the generated tokens.
                        //     // We need to remove the number of padding tokens in the distance computation.
                        //     //   ti   : 0 1 2 3 4 5 6 7 8 9(tlength)
                        //     //   token: i i i i p p p o o o where i=input, p=pad, o=output.
                        //     // e.g. ti = 2, dist = (9 - 3) - 2 = 4.
                        //     float dist = local_time_now - tlength;

                        //     qk_ += mul<float, T, float>(params.linear_bias_slopes[hi], dist);
                        // }

                        // Calculate the max for softmax, and store qk back to smem.
                        // Don't need mask here as we remove paddings in kv cache.
                        qk_max = fmaxf(qk_max, qk_);
                        qk_smem[local_ti] = qk_;
                    }
                }
            }
        }

////////////////////////////////////////////////////////////////////////////////////////////////
// Softmax.

// Perform the final reduction to compute the max inside each warp.
//
// NOTE: In a group of THREADS_PER_KEY threads, the leader already has the max value for the
// group so it's not needed to run the reduction inside the group (again).
#pragma unroll
        for (int mask = WARP_SIZE / 2; mask >= THREADS_PER_KEY; mask /= 2)
        {
            qk_max = fmaxf(qk_max, __shfl_xor_sync(uint32_t(-1), qk_max, mask));
        }

        // Decompose the thread index into warp and lane.
        const auto warp = tidx / WARP_SIZE;
        const auto lane = tidx % WARP_SIZE;

        // The warp leader writes the max to shared memory.
        if (lane == 0)
        {
            red_smem[warp] = qk_max;
        }

        // Make sure the products are in shared memory.
        __syncthreads();

        // The warps finalize the reduction.
        qk_max = lane < WARPS_PER_BLOCK ? red_smem[lane] : -FLT_MAX;
#pragma unroll
        for (int mask = WARPS_PER_BLOCK / 2; mask >= 1; mask /= 2)
        {
            qk_max = fmaxf(qk_max, __shfl_xor_sync(uint32_t(-1), qk_max, mask));
        }

        // Broadcast to all the threads in the warp.
        qk_max = __shfl_sync(uint32_t(-1), qk_max, 0);

        // Compute the logits and start the sum.
        float sum = 0.f;

        // Each thread will handle one float (either qk_smem/logit).
        // const int logit_loop_end = MULTI_BLOCK_FLAG ? timesteps_per_block : tlength;
        const int logit_loop_end = tlength;
        for (int ti = tidx; ti <= logit_loop_end; ti += THREADS_PER_BLOCK)
        {

            // const int time_now = MULTI_BLOCK_FLAG ? ti + c_tile_times_timesteps_per_block : ti;
            const int time_now = ti;

            // For single-block mode, we don't need the mask since it has been skipped.
            // if (!MULTI_BLOCK_FLAG)
            {
                float logit = __expf(qk_smem[time_now] - qk_max);
                sum += logit;
                qk_smem[time_now] = logit;
            }
        }

        // Compute the sum.
        sum = block_sum<WARPS_PER_BLOCK>(&red_smem[WARPS_PER_BLOCK], sum);

        // Normalize the logits.
        float inv_sum = __fdividef(1.f, sum + 1.e-6f);

        // const int normlization_loop_end = MULTI_BLOCK_FLAG ? timesteps_per_block : tlength;
        const int normlization_loop_end = tlength;
        for (int ti = tidx; ti <= normlization_loop_end; ti += THREADS_PER_BLOCK)
        {

            // const int time_now = MULTI_BLOCK_FLAG ? ti + c_tile_times_timesteps_per_block : ti;
            const int time_now = ti;

            // if (!MULTI_BLOCK_FLAG)
            {
                convert_from_float(&logits_smem[ti], qk_smem[ti] * inv_sum);
            }
            // else
            // {
            //     // no scaling factor inv_sum applied here, will apply the scaling factor after all blocks finished
            //     if (time_now < tlength && ti != timesteps_per_block)
            //     {
            //         convert_from_float(&logits_smem[ti], qk_smem[ti]);
            //     }
            //     else if (time_now == tlength)
            //     {
            //         convert_from_float(&logits_current_smem[0], qk_current_smem[0]);
            //     }
            // }
        }

        // Put Values part below so we leverage __syncthreads
        // from the previous step
        // Base pointer for the row of pointers to v cache blocks
        void **v_cache_base_row_ptr = reinterpret_cast<void **>(kvCacheBuffer.getRowPtr(KVIdxType::V_IDX, bi));
        // Base pointer for the row of pointers to v cache blocks for beam's batch, before offsetting with indirection
        // buffer
        void **v_cache_batch_row_ptr = reinterpret_cast<void **>(kvCacheBuffer.getRowPtr(KVIdxType::V_IDX, bi));// / beam_width * beam_width));

        // The number of values processed per iteration of the loop.
        constexpr unsigned V_PER_ITER{THREADS_PER_BLOCK / THREADS_PER_VALUE};
        // The number of unrolled keys per ieration.
        constexpr unsigned UNROLLED_V_PER_ITER = V_PER_ITER * V_LOOP_UNROLL;

        bool const is_valid_vi = IS_Dh_MAX || vi < Dh;

        // One group of threads computes the product(s) for the current timestep.
        // V_vec_k v_bias;
        // zero(v_bias);
        // // if( vo == params.timestep % V_PER_ITER ) {
        // if (is_valid_vi && handle_kv && vo == tlength % V_PER_ITER)
        // {
        //     // Trigger the loads from the V bias buffer.
        //     if (params.v_bias != nullptr)
        //     {
        //         const auto v_bias_offset = flat_index2(hi_kv, vi, Dh);
        //         v_bias = *reinterpret_cast<const V_vec_k *>(&params.v_bias[v_bias_offset]);
        //     }
        // }

        // From previous, before values, step
        // Also make sure the logits are in shared memory.
        __syncthreads();

        ////////////////////////////////////////////////////////////////////////////////////////////////
        // Value cache loops.
#ifdef MMHA_USE_FP32_ACUM_FOR_OUT
        using V_vec_acum = typename V_vec_acum_fp32_<V_vec_k>::Type;
#else
        using V_vec_acum = V_vec_k;
#endif
        // The partial outputs computed by each thread.
        V_vec_acum out;
        zero(out);

        // Loop over the timesteps to compute the partial outputs.
        if (is_valid_vi)
        {
            // Handle only context value cache with beam searching.
            // Handle both context and generation value cache without beam searching.
            // Explict batching of LDGs (by V_LOOP_UNROLL) as it doesn't depend on indirection tables.
            // Take all previous cache as context when we have no beam searching in order to batch as many LDGs as possible.
            const int context_length = tlength;  // const int context_length = HAS_BEAMS ? input_length : tlength;
            int context_v_loop_end = context_length; // int context_v_loop_end = MULTI_BLOCK_FLAG ? timesteps_per_block : context_length;
            int generation_v_loop_end = tlength;  //  int generation_v_loop_end = MULTI_BLOCK_FLAG ? timesteps_per_block : tlength;
            if constexpr (ENABLE_ZEROS)
            {
                for (int ti = vo; ti < context_v_loop_end; ti += UNROLLED_V_PER_ITER)
                {
                    V_vec_m v_vec_cache[V_LOOP_UNROLL];
                    float v_scale_quant_orig_local[V_LOOP_UNROLL];
                    float v_zeros_local[V_LOOP_UNROLL];
#pragma unroll
                    for (int v_loop = 0; v_loop < V_LOOP_UNROLL; v_loop++)
                    {
                        // Fetch offset based on cache_indir when beam sampling
                        int time_idx = ti + v_loop * V_PER_ITER;    // int time_idx = ti + v_loop * V_PER_ITER + (MULTI_BLOCK_FLAG ? c_tile_times_timesteps_per_block : 0);
                        // Haotian: we probably do not need this because each page also contains slots for OOB tokens
                        // time_idx = min(time_idx, tlength - 1);
                        int rowIdx = bi / beam_width * beam_width;

                        const int inBlockIdx = kvCacheBuffer.getKVLocalIdx(time_idx, hi_kv, Dh / (ENABLE_4BITS_CACHE ? 2 : 1), vi / (ENABLE_4BITS_CACHE ? 2 : 1));
                        // The base pointer for the value in the cache buffer.
                        Tcache *v_cache_batch = reinterpret_cast<Tcache *>(kvCacheBuffer.getVBlockPtr(rowIdx, time_idx));

                        half *v_scale_quant_orig_local_ptr = reinterpret_cast<half *>(v_cache_batch + kvCacheBuffer.mBytesPerSeq);
                        int v_scale_quant_orig_local_index = hi_kv * kvCacheBuffer.mTokensPerBlock + kvCacheBuffer.getLocalIdx(time_idx);
                        half *v_zeros_local_ptr = v_scale_quant_orig_local_ptr + kvCacheBuffer.mTokensPerBlock * num_heads_kv;

                        v_vec_cache[v_loop] = *reinterpret_cast<const V_vec_m *>(&v_cache_batch[inBlockIdx]);
                        if constexpr (SMEM_PRELOAD)
                        {
                            v_scale_quant_orig_local[v_loop] = v_scales_history_smem[time_idx];//__half2float(v_scale_quant_orig_local_ptr[v_scale_quant_orig_local_index]);
                            v_zeros_local[v_loop] = v_zeros_history_smem[time_idx];//__half2float(v_zeros_local_ptr[v_scale_quant_orig_local_index]);
                        }
                        else
                        {
                            v_scale_quant_orig_local[v_loop] = __half2float(v_scale_quant_orig_local_ptr[v_scale_quant_orig_local_index]);
                            v_zeros_local[v_loop] = __half2float(v_zeros_local_ptr[v_scale_quant_orig_local_index]);   
                        }
                    }

#pragma unroll
                    for (int v_loop = 0; v_loop < V_LOOP_UNROLL; v_loop++)
                    {
                        V_vec_k v_vec;
                        // we move quantization to here for better batching of inflight LDGs.
                        if constexpr (ENABLE_4BITS_CACHE)
                        {
                            convert_from_4bit_kv_cache<V_vec_m, V_vec_k, Tcache, T_scale>(
                                &v_vec, v_vec_cache[v_loop], v_scale_quant_orig_local[v_loop], v_zeros_local[v_loop]);
                        }
                        else if constexpr (ENABLE_8BITS_CACHE)
                        {
                            convert_from_8bit_kv_cache<V_vec_m, V_vec_k, Tcache, T_scale>(
                                &v_vec, v_vec_cache[v_loop], v_scale_quant_orig_local[v_loop], v_zeros_local[v_loop]);
                        }
                        else
                        {
                            // V_vek is same as V_vec_cache in this case.
                            v_vec = *reinterpret_cast<V_vec_k *>(&v_vec_cache[v_loop]);
                        }

                        int local_time_idx = ti + v_loop * V_PER_ITER;
                        int time_idx = local_time_idx ; //+ (MULTI_BLOCK_FLAG ? c_tile_times_timesteps_per_block : 0);

                        // const bool is_mask = (MULTI_BLOCK_FLAG && local_time_idx >= timesteps_per_block) || (time_idx >= context_length);
                        const bool is_mask = (time_idx >= tlength);
                        // Load the logits from shared memory.
                        if (!is_mask)
                        {
#if defined(MMHA_USE_FP32_ACUM_FOR_LOGITS)
                            float logit = logits_smem[local_time_idx];
                            out = fma(logit, cast_to_float(v_vec), out);
#else  // MMHA_USE_FP32_ACUM_FOR_LOGITS
                            Tk logit = logits_smem[local_time_idx];
                            out = fma(logit, v_vec, out);
#endif // MMHA_USE_FP32_ACUM_FOR_LOGITS
                        }
                    }
                }
            }
            else
            {
                for (int ti = vo; ti < context_v_loop_end; ti += UNROLLED_V_PER_ITER)
                {
                    V_vec_m v_vec_cache[V_LOOP_UNROLL];
                    float v_scale_quant_orig_local[V_LOOP_UNROLL];
#pragma unroll
                    for (int v_loop = 0; v_loop < V_LOOP_UNROLL; v_loop++)
                    {
                        // Fetch offset based on cache_indir when beam sampling
                        int time_idx = ti + v_loop * V_PER_ITER; // + (MULTI_BLOCK_FLAG ? c_tile_times_timesteps_per_block : 0);
                        time_idx = min(time_idx, tlength - 1);
                        int rowIdx = bi / beam_width * beam_width;

                        const int inBlockIdx = kvCacheBuffer.getKVLocalIdx(time_idx, hi_kv, Dh / (ENABLE_4BITS_CACHE ? 2 : 1), vi / (ENABLE_4BITS_CACHE ? 2 : 1));
                        // The base pointer for the value in the cache buffer.
                        Tcache *v_cache_batch = reinterpret_cast<Tcache *>(kvCacheBuffer.getVBlockPtr(rowIdx, time_idx));

                        half *v_scale_quant_orig_local_ptr = reinterpret_cast<half *>(v_cache_batch + kvCacheBuffer.mBytesPerSeq);
                        int v_scale_quant_orig_local_index = kvCacheBuffer.getLocalIdx(time_idx) * num_heads_kv + hi_kv;

                        v_vec_cache[v_loop] = *reinterpret_cast<const V_vec_m *>(&v_cache_batch[inBlockIdx]);
                        // v_scale_quant_orig_local[v_loop] = __half2float(params.v_scale_quant_orig[(rowIdx * max_seq_len + time_idx) * num_heads_kv + hi_kv]);
                        v_scale_quant_orig_local[v_loop] = __half2float(v_scale_quant_orig_local_ptr[v_scale_quant_orig_local_index]);
                    }

#pragma unroll
                    for (int v_loop = 0; v_loop < V_LOOP_UNROLL; v_loop++)
                    {
                        V_vec_k v_vec;
                        // we move quantization to here for better batching of inflight LDGs.
                        if constexpr (ENABLE_4BITS_CACHE)
                        {
                            convert_from_4bit_kv_cache<V_vec_m, V_vec_k, Tcache, T_scale>(
                                &v_vec, v_vec_cache[v_loop], v_scale_quant_orig_local[v_loop]);
                        }
                        else if constexpr (ENABLE_8BITS_CACHE)
                        {
                            convert_from_8bit_kv_cache<V_vec_m, V_vec_k, Tcache, T_scale>(
                                &v_vec, v_vec_cache[v_loop], v_scale_quant_orig_local[v_loop]);
                        }
                        else
                        {
                            // V_vek is same as V_vec_cache in this case.
                            v_vec = *reinterpret_cast<V_vec_k *>(&v_vec_cache[v_loop]);
                        }

                        int local_time_idx = ti + v_loop * V_PER_ITER;
                        int time_idx = local_time_idx; // + (MULTI_BLOCK_FLAG ? c_tile_times_timesteps_per_block : 0);

                        // const bool is_mask = (MULTI_BLOCK_FLAG && local_time_idx >= timesteps_per_block) || (time_idx >= context_length);
                        const bool is_mask = (time_idx >= context_length);
                        // Load the logits from shared memory.
                        if (!is_mask)
                        {
#if defined(MMHA_USE_FP32_ACUM_FOR_LOGITS)
                            float logit = logits_smem[local_time_idx];
                            out = fma(logit, cast_to_float(v_vec), out);
#else  // MMHA_USE_FP32_ACUM_FOR_LOGITS
                            Tk logit = logits_smem[local_time_idx];
                            out = fma(logit, v_vec, out);
#endif // MMHA_USE_FP32_ACUM_FOR_LOGITS
                        }
                    }
                }
            }
        }

        // One group of threads computes the product(s) for the current timestep.
        // if (vo == tlength % V_PER_ITER && is_valid_vi && (!MULTI_BLOCK_FLAG || (c_tile == gridDim.z - 1)))
        if (vo == tlength % V_PER_ITER && is_valid_vi && (c_tile == gridDim.z - 1))
        {
            const int tokenIdx = tlength;
            const int inBlockIdx = kvCacheBuffer.getKVLocalIdx(tokenIdx, hi_kv, Dh / (ENABLE_4BITS_CACHE ? 2 : 1), vi / (ENABLE_4BITS_CACHE ? 2 : 1));
            // The base pointer for the value in the cache buffer.
            Tcache *v_cache_base = reinterpret_cast<Tcache *>(kvCacheBuffer.getBlockPtr(v_cache_base_row_ptr, tokenIdx));

            V_vec_k v;
            {
                // Removed DO_CROSS_ATTENTION branch
                // Trigger the loads from the V buffer.
                // The stride between tokens. We may be able to always use params.stride.
                uint32_t v_stride = params.stride ? static_cast<uint32_t>(params.stride) : (num_heads_kv * Dh);
                // The offset.
                const auto v_offset = flat_index_strided3(bi, hi_kv, vi, v_stride, Dh);
                {
                    // Removed a branch for load_qkv_quant (current step qkv)
                    v = *reinterpret_cast<const V_vec_k *>(&params.v[v_offset]);
                }
            }
            
            // exit early to check OOM
            // PASSED
            // return;

            // if (handle_kv)
            // {
            //     // Compute the V values with bias.
            //     v = add(v, v_bias);

            //     if (do_ia3)
            //     {
            //         v = mul<V_vec_k, V_vec_k, V_vec_k>(v,
            //                                            *reinterpret_cast<const V_vec_k *>(
            //                                                &params.ia3_value_weights[flat_index2(ia3_ti_hi, vi, Dh)]));
            //     }
            // }

            // Store the values with bias back to global memory in the cache for V.
            //*reinterpret_cast<V_vec_k*>(&v_cache[params.timestep*Dh]) = v;
            // For MQA/GQA mode, write only with the first Q head of each group per KV head.
            if (hi == (hi_kv * qhead_per_kv))
            {
                if constexpr (ENABLE_ZEROS)
                {
                    if constexpr (ENABLE_4BITS_CACHE)
                    {
                        store_4bits_kv_cache_vec(v_cache_base, v, inBlockIdx, v_scale_orig_quant, v_zeros);
                    }
                    else if constexpr (ENABLE_8BITS_CACHE)
                    {
                        store_8bits_kv_cache_vec(v_cache_base, v, inBlockIdx, v_scale_orig_quant, v_zeros);
                    }
                    else
                    {
                        *reinterpret_cast<V_vec_k *>(&v_cache_base[inBlockIdx]) = v;
                    }
                }
                else
                {
                    if constexpr (ENABLE_4BITS_CACHE)
                    {
                        store_4bits_kv_cache_vec(v_cache_base, v, inBlockIdx, v_scale_orig_quant);
                    }
                    else if constexpr (ENABLE_8BITS_CACHE)
                    {
                        store_8bits_kv_cache_vec(v_cache_base, v, inBlockIdx, v_scale_orig_quant);
                    }
                    else
                    {
                        *reinterpret_cast<V_vec_k *>(&v_cache_base[inBlockIdx]) = v;
                    }
                }
            }

            if constexpr (ENABLE_ZEROS && ENABLE_4BITS_CACHE)
            {
                v = reorder_8xfp16(v);
            }

            // exit early to check OOM
            // FAILED
            // return;

            // Initialize the output value with the current timestep.
#if defined(MMHA_USE_FP32_ACUM_FOR_LOGITS)
            // out = fma(logits_smem[params.timestep], cast_to_float(v), out);
            // if (!MULTI_BLOCK_FLAG)
            {
                out = fma(logits_smem[tlength], cast_to_float(v), out);
            }
            // else
            // {
            //     out = fma(logits_current_smem[0], cast_to_float(v), out);
            // }
#else  // MMHA_USE_FP32_ACUM_FOR_LOGITS
       // out = fma(logits_smem[params.timestep], v, out);
            // if (!MULTI_BLOCK_FLAG)
            {
                out = fma(logits_smem[tlength], v, out);
            }
            // else
            // { // MULTI_BLOCK_FLAG // Not supported yet: multi-block mode with FP8_MHA
            //     out = fma(logits_current_smem[0], v, out);
            // }
#endif // MMHA_USE_FP32_ACUM_FOR_LOGITS
        }
        // exit early to check OOM
        // FAILED
        // return;

        // Make sure we can start writing to shared memory.
        __syncthreads();

        // Run the final reduction amongst the different groups computing different partial outputs.
#pragma unroll
        for (int active_groups = V_PER_ITER; active_groups >= 2; active_groups /= 2)
        {

            // The midpoint in the number of active groups.
            int midpoint = active_groups / 2;

            // The upper part of active threads store to shared memory.
            if (vo >= midpoint && vo < active_groups && (Dh == Dh_MAX || vi < Dh))
            {
#ifdef MMHA_USE_FP32_ACUM_FOR_OUT
                convert_from_float(reinterpret_cast<V_vec_k *>(&out_smem[(vo - midpoint) * Dh + vi]), out);
#else
                *reinterpret_cast<V_vec_k *>(&out_smem[(vo - midpoint) * Dh + vi]) = out;
#endif
            }
            __syncthreads();

            // The bottom warps update their values.
            if (vo < midpoint && (Dh == Dh_MAX || vi < Dh))
            {
                out = add(*reinterpret_cast<const V_vec_k *>(&out_smem[vo * Dh + vi]), out);
            }
            __syncthreads();
        }

        const auto bhi = flat_index2(bi, hi, num_heads);
        const auto bhi_seq_len_tile = bhi * params.max_seq_len_tile;
        // Output the final values.
        if (vo == 0 && (Dh == Dh_MAX || vi < Dh))
        {
            const auto bhvi = flat_index2(bhi, vi, Dh);
#ifdef MMHA_USE_FP32_ACUM_FOR_OUT
            if (write_attention_quant)
            {
                using Packed_Int8_t = typename packed_type<int8_t, num_elems<V_vec_acum>::value>::type;
                out = mul<V_vec_acum, float>(*params.attention_out_scale_orig_quant, out);
                *reinterpret_cast<Packed_Int8_t *>(&(reinterpret_cast<int8_t *>(params.out)[bhvi])) = cast_to_int8(out);
            }
            else
            {
                // if (!MULTI_BLOCK_FLAG)
                {
                    // This makes sure we have coalesced memory access.
                    V_vec_k final_out;
                    convert_from_float(&final_out, out);
                    // 01234567->02461357
                    // TODO (Haotian): correctness in multi-block mode
                    if constexpr (ENABLE_4BITS_CACHE && ENABLE_ZEROS)
                    {
                        final_out = reorder_8xfp16_type2(final_out);
                    }
                    *reinterpret_cast<V_vec_k *>(&params.out[bhvi]) = final_out;
                }
            }
#else  // MMHA_USE_FP32_ACUM_FOR_OUT
            *reinterpret_cast<V_vec_acum *>(&params.out[bhvi]) = out;
#endif // MMHA_USE_FP32_ACUM_FOR_OUT
        }
    }

    template <typename T, int Dh, bool DO_MULTI_BLOCK>
    inline size_t smem_size_in_bytes(const Multihead_attention_params<T> &params, int threads_per_block)
    {
        using Tk = typename kernel_type_t<T>::Type;
        // The amount of shared memory needed to store the Q*K^T values in float.
        // const int max_timesteps = DO_CROSS_ATTENTION
        //                               ? params.memory_max_len
        //                               : min((DO_MULTI_BLOCK ? params.timesteps_per_block : params.timestep), params.memory_max_len);
        const int max_timesteps = min((DO_MULTI_BLOCK ? params.timesteps_per_block : params.timestep), params.memory_max_len);
        const auto qk_elts = static_cast<std::size_t>(divUp(max_timesteps + 1, 4)); // explicit cast because of the sign
        const auto qk_sz = qk_elts * 16;

        // The extra memory needed if we are not using floats for the final logits.
        size_t logits_sz = 0;
#ifndef MMHA_USE_FP32_ACUM_FOR_LOGITS
        if (sizeof(Tk) != 4)
        {
            // TDOD
            logits_sz = qk_elts * 4 * sizeof(Tk);
        }
#endif

        // The total size needed during softmax.
        size_t softmax_sz = qk_sz + logits_sz;

        auto constexpr threads_per_value = mmha::threads_per_value<T>(dh_max(Dh));

        // The number of partial rows to reduce in the final reduction.
        int rows_per_red = threads_per_block / threads_per_value;
        // The amount of storage needed to finalize the outputs.
        size_t red_sz = rows_per_red * params.hidden_size_per_head * sizeof(Tk) / 2;

        size_t transpose_rotary_size = 0;
        if (params.position_embedding_type == PositionEmbeddingType::kROPE_GPT_NEOX)
        {
            // assert(params.rotary_embedding_dim > 0);
            transpose_rotary_size = 2 * params.rotary_embedding_dim * sizeof(Tk);
        }

        size_t out_oi_sz = 0;
        if (params.multi_block_mode)
        {
            // The size for partial output reduction computation.
            out_oi_sz = params.max_seq_len_tile * params.hidden_size_per_head * sizeof(T);
        }

        // The max.
        return max(max(max(softmax_sz, red_sz), transpose_rotary_size), out_oi_sz);
    }

} // namespace mmha