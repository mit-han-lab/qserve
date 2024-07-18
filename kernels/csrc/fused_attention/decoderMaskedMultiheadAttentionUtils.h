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
#include "gptKernels.h"
#include <stdint.h>

namespace mmha
{
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ constexpr uint32_t shfl_mask(int threads)
    {
        assert(threads <= 32);
        return threads == 32 ? -1u : (1u << threads) - 1u;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    struct __align__(16) Float4_
    {
        float2 x;
        float2 y;
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    struct __align__(32) Float8_
    {
        float2 x;
        float2 y;
        float2 z;
        float2 w;
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <typename T>
    struct num_elems;

    template <>
    struct num_elems<float>
    {
        static constexpr int value = 1;
    };

    template <>
    struct num_elems<float2>
    {
        static constexpr int value = 2;
    };

    template <>
    struct num_elems<float4>
    {
        static constexpr int value = 4;
    };

    template <>
    struct num_elems<Float4_>
    {
        static constexpr int value = 4;
    };

    template <>
    struct num_elems<Float8_>
    {
        static constexpr int value = 8;
    };

    template <>
    struct num_elems<half>
    {
        static constexpr int value = 1;
    };

    template <>
    struct num_elems<uint32_t>
    {
        static constexpr int value = 2;
    };

    template <>
    struct num_elems<uint2>
    {
        static constexpr int value = 4;
    };

    template <>
    struct num_elems<uint4>
    {
        static constexpr int value = 8;
    };


    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <typename T, int N>
    struct packed_type;

    template <typename T>
    struct packed_type<T, 1>
    {
        using type = T;
    };

    template <>
    struct packed_type<int8_t, 1>
    {
        using type = int8_t;
    };

    template <>
    struct packed_type<int8_t, 2>
    {
        using type = int16_t;
    };

    template <>
    struct packed_type<int8_t, 4>
    {
        using type = int32_t;
    };

    template <>
    struct packed_type<int8_t, 8>
    {
        using type = int64_t;
    };

    template <>
    struct packed_type<uint16_t, 2>
    {
        using type = uint32_t;
    };

    template <>
    struct packed_type<uint16_t, 4>
    {
        using type = uint2;
    };

    template <>
    struct packed_type<uint16_t, 8>
    {
        using type = uint4;
    };

    template <>
    struct packed_type<half, 2>
    {
        using type = uint32_t;
    };

    template <>
    struct packed_type<half, 4>
    {
        using type = uint2;
    };

    template <>
    struct packed_type<half, 8>
    {
        using type = uint4;
    };

    template <>
    struct packed_type<float, 2>
    {
        using type = float2;
    };

    template <>
    struct packed_type<float, 4>
    {
        using type = float4;
    };

    template <>
    struct packed_type<float, 8>
    {
        using type = Float8_;
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ float add(float a, float b)
    {
        return a + b;
    }

    inline __device__ float sub(float a, float b)
    {
        return a - b;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ float2 add(float2 a, float2 b)
    {
        float2 c;
        c.x = add(a.x, b.x);
        c.y = add(a.y, b.y);
        return c;
    }

    inline __device__ float2 add(float2 a, float b)
    {
        float2 c;
        c.x = add(a.x, b);
        c.y = add(a.y, b);
        return c;
    }

    inline __device__ float2 sub(float2 a, float b)
    {
        float2 c;
        c.x = sub(a.x, b);
        c.y = sub(a.y, b);
        return c;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ float4 add(float4 a, float4 b)
    {
        float4 c;
        c.x = add(a.x, b.x);
        c.y = add(a.y, b.y);
        c.z = add(a.z, b.z);
        c.w = add(a.w, b.w);
        return c;
    }

    inline __device__ float4 add(float4 a, float b)
    {
        float4 c;
        c.x = add(a.x, b);
        c.y = add(a.y, b);
        c.z = add(a.z, b);
        c.w = add(a.w, b);
        return c;
    }

    inline __device__ float4 sub(float4 a, float b)
    {
        float4 c;
        c.x = sub(a.x, b);
        c.y = sub(a.y, b);
        c.z = sub(a.z, b);
        c.w = sub(a.w, b);
        return c;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ Float8_ add(Float8_ a, Float8_ b)
    {
        Float8_ c;
        c.x = add(a.x, b.x);
        c.y = add(a.y, b.y);
        c.z = add(a.z, b.z);
        c.w = add(a.w, b.w);
        return c;
    }

    inline __device__ Float8_ add(Float8_ a, float b)
    {
        Float8_ c;
        c.x = add(a.x, b);
        c.y = add(a.y, b);
        c.z = add(a.z, b);
        c.w = add(a.w, b);
        return c;
    }

    inline __device__ Float8_ sub(Float8_ a, float b)
    {
        Float8_ c;
        c.x = sub(a.x, b);
        c.y = sub(a.y, b);
        c.z = sub(a.z, b);
        c.w = sub(a.w, b);
        return c;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ uint16_t add(uint16_t a, uint16_t b)
    {
        uint16_t c;
        asm volatile("add.f16 %0, %1, %2;\n" : "=h"(c) : "h"(a), "h"(b));
        return c;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ uint32_t add(uint32_t a, uint32_t b)
    {
        uint32_t c;
        asm volatile("add.f16x2 %0, %1, %2;\n" : "=r"(c) : "r"(a), "r"(b));
        return c;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ uint2 add(uint2 a, uint2 b)
    {
        uint2 c;
        c.x = add(a.x, b.x);
        c.y = add(a.y, b.y);
        return c;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ uint4 add(uint4 a, uint4 b)
    {
        uint4 c;
        c.x = add(a.x, b.x);
        c.y = add(a.y, b.y);
        c.z = add(a.z, b.z);
        c.w = add(a.w, b.w);
        return c;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ uint16_t float_to_half(float f)
    {
        union
        {
            uint32_t u32;
            uint16_t u16[2];
        } tmp;
#if 0 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 // Is it better?
    float zero = 0.f;
    asm volatile("cvt.rn.f16x2.f32 %0, %1, %2;\n" : "=r"(tmp.u32) : "f"(zero), "f"(f));
#else
        asm volatile("cvt.rn.f16.f32 %0, %1;\n" : "=h"(tmp.u16[0]) : "f"(f));
#endif
        return tmp.u16[0];
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ uint32_t float2_to_half2(float2 f)
    {
        union
        {
            uint32_t u32;
            uint16_t u16[2];
        } tmp;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        asm volatile("cvt.rn.f16x2.f32 %0, %1, %2;\n" : "=r"(tmp.u32) : "f"(f.y), "f"(f.x));
#else
        asm volatile("cvt.rn.f16.f32 %0, %1;\n" : "=h"(tmp.u16[0]) : "f"(f.x));
        asm volatile("cvt.rn.f16.f32 %0, %1;\n" : "=h"(tmp.u16[1]) : "f"(f.y));
#endif
        return tmp.u32;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ float half_to_float(uint16_t h)
    {
        float f;
        asm volatile("cvt.f32.f16 %0, %1;\n" : "=f"(f) : "h"(h));
        return f;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ float2 half2_to_float2(uint32_t v)
    {
        uint16_t lo, hi;
        asm volatile("mov.b32 {%0, %1}, %2;\n" : "=h"(lo), "=h"(hi) : "r"(v));
        return make_float2(half_to_float(lo), half_to_float(hi));
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ float add(float a, uint16_t b)
    {
        return a + half_to_float(b);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ float2 add(uint32_t a, float2 fb)
    {
        float2 fa = half2_to_float2(a);
        return add(fa, fb);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ Float4_ add(uint2 a, Float4_ fb)
    {
        Float4_ fc;
        fc.x = add(a.x, fb.x);
        fc.y = add(a.y, fb.y);
        return fc;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ Float8_ add(uint4 a, Float8_ fb)
    {
        Float8_ fc;
        fc.x = add(a.x, fb.x);
        fc.y = add(a.y, fb.y);
        fc.z = add(a.z, fb.z);
        fc.w = add(a.w, fb.w);
        return fc;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ uint32_t h0_h0(uint16_t a)
    {
        uint32_t b;
        asm volatile("mov.b32 %0, {%1, %1};" : "=r"(b) : "h"(a));
        return b;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ float fma(float a, float b, float c)
    {
        return a * b + c;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ float2 fma(float2 a, float2 b, float2 c)
    {
        float2 d;
        d.x = fma(a.x, b.x, c.x);
        d.y = fma(a.y, b.y, c.y);
        return d;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ float2 fma(float a, float2 b, float2 c)
    {
        float2 d;
        d.x = fma(a, b.x, c.x);
        d.y = fma(a, b.y, c.y);
        return d;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ float4 fma(float4 a, float4 b, float4 c)
    {
        float4 d;
        d.x = fma(a.x, b.x, c.x);
        d.y = fma(a.y, b.y, c.y);
        d.z = fma(a.z, b.z, c.z);
        d.w = fma(a.w, b.w, c.w);
        return d;
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ Float8_ fma(Float8_ a, Float8_ b, Float8_ c)
    {
        Float8_ d;
        d.x = fma(a.x, b.x, c.x);
        d.y = fma(a.y, b.y, c.y);
        d.z = fma(a.z, b.z, c.z);
        d.w = fma(a.w, b.w, c.w);
        return d;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ float4 fma(float a, float4 b, float4 c)
    {
        float4 d;
        d.x = fma(a, b.x, c.x);
        d.y = fma(a, b.y, c.y);
        d.z = fma(a, b.z, c.z);
        d.w = fma(a, b.w, c.w);
        return d;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ float4 fma(float a, float4 b, Float4_ c)
    {
        float4 d;
        d.x = fma(a, b.x, c.x.x);
        d.y = fma(a, b.y, c.x.y);
        d.z = fma(a, b.z, c.y.x);
        d.w = fma(a, b.w, c.y.y);
        return d;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ Float4_ fma(float a, Float4_ b, Float4_ c)
    {
        Float4_ d;
        d.x = fma(a, b.x, c.x);
        d.y = fma(a, b.y, c.y);
        return d;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ Float8_ fma(float a, Float8_ b, Float8_ c)
    {
        Float8_ d;
        d.x = fma(a, b.x, c.x);
        d.y = fma(a, b.y, c.y);
        d.z = fma(a, b.z, c.z);
        d.w = fma(a, b.w, c.w);
        return d;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ uint32_t fma(uint32_t a, uint32_t b, uint32_t c)
    {
        uint32_t d;
        asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(c));
        return d;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ uint32_t fma(uint16_t a, uint32_t b, uint32_t c)
    {
        return fma(h0_h0(a), b, c);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ uint2 fma(uint2 a, uint2 b, uint2 c)
    {
        uint2 d;
        d.x = fma(a.x, b.x, c.x);
        d.y = fma(a.y, b.y, c.y);
        return d;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ uint2 fma(uint16_t a, uint2 b, uint2 c)
    {
        uint32_t s = h0_h0(a);
        uint2 d;
        d.x = fma(s, b.x, c.x);
        d.y = fma(s, b.y, c.y);
        return d;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ uint4 fma(uint4 a, uint4 b, uint4 c)
    {
        uint4 d;
        d.x = fma(a.x, b.x, c.x);
        d.y = fma(a.y, b.y, c.y);
        d.z = fma(a.z, b.z, c.z);
        d.w = fma(a.w, b.w, c.w);
        return d;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ uint4 fma(uint16_t a, uint4 b, uint4 c)
    {
        uint32_t s = h0_h0(a);
        uint4 d;
        d.x = fma(s, b.x, c.x);
        d.y = fma(s, b.y, c.y);
        d.z = fma(s, b.z, c.z);
        d.w = fma(s, b.w, c.w);
        return d;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ float fma(uint16_t a, uint16_t b, float fc)
    {
        float fa = half_to_float(a);
        float fb = half_to_float(b);
        return fa * fb + fc;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ float2 fma(uint32_t a, uint32_t b, float2 fc)
    {
        float2 fa = half2_to_float2(a);
        float2 fb = half2_to_float2(b);
        return fma(fa, fb, fc);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ float2 fma(uint16_t a, uint32_t b, float2 fc)
    {
        return fma(h0_h0(a), b, fc);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ Float4_ fma(uint2 a, uint2 b, Float4_ fc)
    {
        Float4_ fd;
        fd.x = fma(a.x, b.x, fc.x);
        fd.y = fma(a.y, b.y, fc.y);
        return fd;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ Float4_ fma(uint16_t a, uint2 b, Float4_ fc)
    {
        uint32_t s = h0_h0(a);
        Float4_ fd;
        fd.x = fma(s, b.x, fc.x);
        fd.y = fma(s, b.y, fc.y);
        return fd;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ Float8_ fma(uint4 a, uint4 b, Float8_ fc)
    {
        Float8_ fd;
        fd.x = fma(a.x, b.x, fc.x);
        fd.y = fma(a.y, b.y, fc.y);
        fd.z = fma(a.z, b.z, fc.z);
        fd.w = fma(a.w, b.w, fc.w);
        return fd;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ Float8_ fma(uint16_t a, uint4 b, Float8_ fc)
    {
        uint32_t s = h0_h0(a);
        Float8_ fd;
        fd.x = fma(s, b.x, fc.x);
        fd.y = fma(s, b.y, fc.y);
        fd.z = fma(s, b.z, fc.z);
        fd.w = fma(s, b.w, fc.w);
        return fd;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ uint32_t sub(uint32_t a, uint32_t b)
    {
        uint32_t c;
        asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(c) : "r"(a), "r"(b));
        return c;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ __nv_bfloat162 sub(__nv_bfloat162 a, __nv_bfloat162 b)
    {
        return hsub2(a, b);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ float2 sub(float2 a, float2 b)
    {
        float2 c;
        c.x = a.x - b.x;
        c.y = a.y - b.y;
        return c;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <typename Acc, typename A, typename B>
    inline __device__ Acc mul(A a, B b)
    {
        return Acc{}; // for compile
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <>
    inline __device__ float mul<float, float>(float a, float b)
    {
        return a * b;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <>
    inline __device__ float2 mul(float2 a, float2 b)
    {
        float2 c;
        c.x = a.x * b.x;
        c.y = a.y * b.y;
        return c;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <>
    inline __device__ float2 mul(float a, float2 b)
    {
        float2 c;
        c.x = a * b.x;
        c.y = a * b.y;
        return c;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <>
    inline __device__ float4 mul(float4 a, float4 b)
    {
        float4 c;
        c.x = a.x * b.x;
        c.y = a.y * b.y;
        c.z = a.z * b.z;
        c.w = a.w * b.w;
        return c;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <>
    inline __device__ float4 mul(float a, float4 b)
    {
        float4 c;
        c.x = a * b.x;
        c.y = a * b.y;
        c.z = a * b.z;
        c.w = a * b.w;
        return c;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <>
    inline __device__ Float8_ mul(float a, Float8_ b)
    {
        Float8_ c;
        c.x = mul<float2, float, float2>(a, b.x);
        c.y = mul<float2, float, float2>(a, b.y);
        c.z = mul<float2, float, float2>(a, b.z);
        c.w = mul<float2, float, float2>(a, b.w);
        return c;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <>
    inline __device__ uint16_t mul(uint16_t a, uint16_t b)
    {
        uint16_t c;
        asm volatile("mul.f16 %0, %1, %2;\n" : "=h"(c) : "h"(a), "h"(b));
        return c;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <>
    inline __device__ uint32_t mul(uint32_t a, uint32_t b)
    {
        uint32_t c;
        asm volatile("mul.f16x2 %0, %1, %2;\n" : "=r"(c) : "r"(a), "r"(b));
        return c;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <>
    inline __device__ uint32_t mul(uint16_t a, uint32_t b)
    {
        return mul<uint32_t, uint32_t, uint32_t>(h0_h0(a), b);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <>
    inline __device__ uint2 mul(uint2 a, uint2 b)
    {
        uint2 c;
        c.x = mul<uint32_t, uint32_t, uint32_t>(a.x, b.x);
        c.y = mul<uint32_t, uint32_t, uint32_t>(a.y, b.y);
        return c;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <>
    inline __device__ uint2 mul(uint16_t a, uint2 b)
    {
        uint32_t s = h0_h0(a);
        uint2 c;
        c.x = mul<uint32_t, uint32_t, uint32_t>(s, b.x);
        c.y = mul<uint32_t, uint32_t, uint32_t>(s, b.y);
        return c;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <>
    inline __device__ uint4 mul(uint4 a, uint4 b)
    {
        uint4 c;
        c.x = mul<uint32_t, uint32_t, uint32_t>(a.x, b.x);
        c.y = mul<uint32_t, uint32_t, uint32_t>(a.y, b.y);
        c.z = mul<uint32_t, uint32_t, uint32_t>(a.z, b.z);
        c.w = mul<uint32_t, uint32_t, uint32_t>(a.w, b.w);
        return c;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <>
    inline __device__ uint4 mul(uint16_t a, uint4 b)
    {
        uint32_t s = h0_h0(a);
        uint4 c;
        c.x = mul<uint32_t, uint32_t, uint32_t>(s, b.x);
        c.y = mul<uint32_t, uint32_t, uint32_t>(s, b.y);
        c.z = mul<uint32_t, uint32_t, uint32_t>(s, b.z);
        c.w = mul<uint32_t, uint32_t, uint32_t>(s, b.w);
        return c;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <>
    inline __device__ float mul(uint16_t a, uint16_t b)
    {
        float fa = half_to_float(a);
        float fb = half_to_float(b);
        return fa * fb;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <>
    inline __device__ float mul(uint16_t a, float b)
    {
        return half_to_float(a) * b;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <>
    inline __device__ float2 mul(uint32_t a, uint32_t b)
    {
        float2 fa = half2_to_float2(a);
        float2 fb = half2_to_float2(b);
        return mul<float2, float2, float2>(fa, fb);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <>
    inline __device__ float2 mul(uint16_t a, uint32_t b)
    {
        return mul<float2, uint32_t, uint32_t>(h0_h0(a), b);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <>
    inline __device__ Float4_ mul(uint2 a, uint2 b)
    {
        Float4_ fc;
        fc.x = mul<float2, uint32_t, uint32_t>(a.x, b.x);
        fc.y = mul<float2, uint32_t, uint32_t>(a.y, b.y);
        return fc;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <>
    inline __device__ Float4_ mul(uint16_t a, uint2 b)
    {
        uint32_t s = h0_h0(a);
        Float4_ fc;
        fc.x = mul<float2, uint32_t, uint32_t>(s, b.x);
        fc.y = mul<float2, uint32_t, uint32_t>(s, b.y);
        return fc;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <>
    inline __device__ Float8_ mul(uint4 a, uint4 b)
    {
        Float8_ fc;
        fc.x = mul<float2, uint32_t, uint32_t>(a.x, b.x);
        fc.y = mul<float2, uint32_t, uint32_t>(a.y, b.y);
        fc.z = mul<float2, uint32_t, uint32_t>(a.z, b.z);
        fc.w = mul<float2, uint32_t, uint32_t>(a.w, b.w);
        return fc;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <>
    inline __device__ Float8_ mul(uint16_t a, uint4 b)
    {
        uint32_t s = h0_h0(a);
        Float8_ fc;
        fc.x = mul<float2, uint32_t, uint32_t>(s, b.x);
        fc.y = mul<float2, uint32_t, uint32_t>(s, b.y);
        fc.z = mul<float2, uint32_t, uint32_t>(s, b.z);
        fc.w = mul<float2, uint32_t, uint32_t>(s, b.w);
        return fc;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <>
    inline __device__ Float8_ mul(float a, uint4 b)
    {
        uint16_t h0 = float_to_half(a);
        uint32_t s = h0_h0(h0);
        Float8_ fc;
        fc.x = mul<float2, uint32_t, uint32_t>(s, b.x);
        fc.y = mul<float2, uint32_t, uint32_t>(s, b.y);
        fc.z = mul<float2, uint32_t, uint32_t>(s, b.z);
        fc.w = mul<float2, uint32_t, uint32_t>(s, b.w);
        return fc;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <>
    inline __device__ float2 mul(uint32_t a, float2 fb)
    {
        float2 fa = half2_to_float2(a);
        return mul<float2, float2, float2>(fa, fb);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <>
    inline __device__ float2 mul(float2 fa, uint32_t b)
    {
        float2 fb = half2_to_float2(b);
        return mul<float2, float2, float2>(fa, fb);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ float sum(float v)
    {
        return v;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ float sum(float2 v)
    {
        return v.x + v.y;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ float sum(float4 v)
    {
        return v.x + v.y + v.z + v.w;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ float sum(Float4_ v)
    {
        return v.x.x + v.x.y + v.y.x + v.y.y;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ float sum(Float8_ v)
    {
        float out = 0.f;

        out += sum(v.x);
        out += sum(v.y);
        out += sum(v.z);
        out += sum(v.w);

        return out;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ float sum(uint16_t v)
    {
        return half_to_float(v);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ float sum(uint32_t v)
    {
        float2 tmp = half2_to_float2(v);
        return tmp.x + tmp.y;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ float sum(uint2 v)
    {
        uint32_t c = add(v.x, v.y);
        return sum(c);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ float sum(uint4 v)
    {
#if 1
        uint32_t c = add(v.x, v.y);
        c = add(c, v.z);
        c = add(c, v.w);
#else
        uint32_t c = add(v.x, v.y);
        uint32_t d = add(v.z, v.w);
        c = add(c, d);
#endif
        return sum(c);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <typename T>
    inline __device__ float dot(T a, T b)
    {
        return sum(mul<T, T, T>(a, b));
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <typename A, typename T>
    inline __device__ float dot(T a, T b)
    {
        return sum(mul<A, T, T>(a, b));
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ void zero(uint16_t &dst)
    {
        dst = uint16_t(0);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <typename T>
    inline __device__ void zero(T &dst)
    {
        constexpr int WORDS = sizeof(T) / 4;

        union
        {
            T raw;
            uint32_t words[WORDS];
        } tmp;

#pragma unroll
        for (int ii = 0; ii < WORDS; ++ii)
        {
            tmp.words[ii] = 0u;
        }
        dst = tmp.raw;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ float update_rotary_base(
        const int kv_seq_len, const int max_positions, const int embed_dim, const float base, const float scale)
    {
        const float b = (scale * kv_seq_len / max_positions) - (scale - 1);
        const float p = static_cast<float>(embed_dim) / (embed_dim - 2);
        return base * pow(b, p);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ float2 update_dynamic_scaling_rotary(float base, float scale, const int kv_seq_len,
                                                           const int max_positions, const int embed_dim, const bool dynamic_scaling)
    {
        const float b = kv_seq_len * __fdividef(scale, max_positions) - (scale - 1);
        const float p = __fdividef(embed_dim, embed_dim - 2);
        const float updated_base = dynamic_scaling ? base * __powf(b, p) : base;
        const float updated_scale = dynamic_scaling ? 1.0f : scale;
        return {updated_base, updated_scale};
    }

    inline __device__ void update_rotary_base_n_scale(float &base, float &scale, RotaryScalingType const scale_type,
                                                      const int rot_embed_dim, const int max_positions, const int seq_len)
    {
        // only update the base and/or scale if needed based on scale_type
        if (scale_type == RotaryScalingType::kDYNAMIC)
        {
            if (seq_len > max_positions)
            {
                base = update_rotary_base(seq_len, max_positions, rot_embed_dim, base, scale);
            }
            scale = 1.0f; // scale is only used in base for dynamic scaling
        }
        else if (scale_type == RotaryScalingType::kLINEAR)
        {
            scale = 1.0f / scale;
        }
    }

    inline __device__ float2 rotary_embedding_coefficient(
        const int zid, const int rot_embed_dim, const float base, const float scale, const float t_step)
    {
        const float inv_freq = (t_step * scale) / pow(base, zid / (float)rot_embed_dim);
        return {cos(inv_freq), sin(inv_freq)};
    }

    inline __device__ float2 rotary_embedding_transform(const float2 v, const float2 coef)
    {
        float2 rot_v;
        rot_v.x = coef.x * v.x - coef.y * v.y;
        rot_v.y = coef.x * v.y + coef.y * v.x;
        return rot_v;
    }

    inline __device__ uint32_t rotary_embedding_transform(const uint32_t v, const float2 coef)
    {
        float2 fv = half2_to_float2(v);
        float2 rot_fv = rotary_embedding_transform(fv, coef);
        return float2_to_half2(rot_fv);
    }

    inline __device__ void apply_rotary_embedding(float &q, int zid, int rot_embed_dim, float base, float scale, int t_step)
    {
        return;
    }

    inline __device__ void apply_rotary_embedding(
        float &q, float &k, int zid, int rot_embed_dim, float base, float scale, int t_step)
    {
        return;
    }

    inline __device__ void apply_rotary_embedding(
        float2 &q, int tid, int rot_embed_dim, float base, float scale, int t_step)
    {
        if (2 * tid >= rot_embed_dim)
        {
            return;
        }
        const auto coef = rotary_embedding_coefficient(2 * tid, rot_embed_dim, base, scale, t_step);
        q = rotary_embedding_transform(q, coef);
    }

    inline __device__ void apply_rotary_embedding(
        float2 &q, float2 &k, int tid, int rot_embed_dim, float base, float scale, int t_step)
    {
        if (2 * tid >= rot_embed_dim)
        {
            return;
        }
        const auto coef = rotary_embedding_coefficient(2 * tid, rot_embed_dim, base, scale, t_step);
        q = rotary_embedding_transform(q, coef);
        k = rotary_embedding_transform(k, coef);
    }

    inline __device__ void apply_rotary_embedding(
        float4 &q, int tid, int rot_embed_dim, float base, float scale, int t_step)
    {
        if (4 * tid >= rot_embed_dim)
        {
            return;
        }

        Float4_ &q_ = *reinterpret_cast<Float4_ *>(&q);
        const auto coef0 = rotary_embedding_coefficient(4 * tid, rot_embed_dim, base, scale, t_step);
        q_.x = rotary_embedding_transform(q_.x, coef0);
        const auto coef1 = rotary_embedding_coefficient(4 * tid + 2, rot_embed_dim, base, scale, t_step);
        q_.y = rotary_embedding_transform(q_.y, coef1);
    }

    inline __device__ void apply_rotary_embedding(
        float4 &q, float4 &k, int tid, int rot_embed_dim, float base, float scale, int t_step)
    {
        if (4 * tid >= rot_embed_dim)
        {
            return;
        }

        Float4_ &q_ = *reinterpret_cast<Float4_ *>(&q);
        Float4_ &k_ = *reinterpret_cast<Float4_ *>(&k);
        const auto coef0 = rotary_embedding_coefficient(4 * tid, rot_embed_dim, base, scale, t_step);
        q_.x = rotary_embedding_transform(q_.x, coef0);
        k_.x = rotary_embedding_transform(k_.x, coef0);
        const auto coef1 = rotary_embedding_coefficient(4 * tid + 2, rot_embed_dim, base, scale, t_step);
        q_.y = rotary_embedding_transform(q_.y, coef1);
        k_.y = rotary_embedding_transform(k_.y, coef1);
    }

    inline __device__ void apply_rotary_embedding(
        uint32_t &q, int tid, int rot_embed_dim, float base, float scale, int t_step)
    {
        if (2 * tid >= rot_embed_dim)
        {
            return;
        }
        const auto coef = rotary_embedding_coefficient(2 * tid, rot_embed_dim, base, scale, t_step);
        q = rotary_embedding_transform(q, coef);
    }

    inline __device__ void apply_rotary_embedding(
        uint32_t &q, uint32_t &k, int tid, int rot_embed_dim, float base, float scale, int t_step)
    {
        if (2 * tid >= rot_embed_dim)
        {
            return;
        }
        const auto coef = rotary_embedding_coefficient(2 * tid, rot_embed_dim, base, scale, t_step);
        q = rotary_embedding_transform(q, coef);
        k = rotary_embedding_transform(k, coef);
    }

    inline __device__ void apply_rotary_embedding(half2 &q, int tid, int rot_embed_dim, float base, float scale, int t_step)
    {
        return apply_rotary_embedding(*reinterpret_cast<uint32_t *>(&q), tid, rot_embed_dim, base, scale, t_step);
    }

    inline __device__ void apply_rotary_embedding(
        half2 &q, half2 &k, int tid, int rot_embed_dim, float base, float scale, int t_step)
    {
        return apply_rotary_embedding(
            *reinterpret_cast<uint32_t *>(&q), *reinterpret_cast<uint32_t *>(&k), tid, rot_embed_dim, base, scale, t_step);
    }

    inline __device__ void apply_rotary_embedding(uint2 &q, int tid, int rot_embed_dim, float base, float scale, int t_step)
    {
        if (4 * tid >= rot_embed_dim)
        {
            return;
        }
        const auto coef0 = rotary_embedding_coefficient(4 * tid, rot_embed_dim, base, scale, t_step);
        q.x = rotary_embedding_transform(q.x, coef0);
        const auto coef1 = rotary_embedding_coefficient(4 * tid + 2, rot_embed_dim, base, scale, t_step);
        q.y = rotary_embedding_transform(q.y, coef1);
    }

    inline __device__ void apply_rotary_embedding(
        uint2 &q, uint2 &k, int tid, int rot_embed_dim, float base, float scale, int t_step)
    {
        if (4 * tid >= rot_embed_dim)
        {
            return;
        }
        const auto coef0 = rotary_embedding_coefficient(4 * tid, rot_embed_dim, base, scale, t_step);
        q.x = rotary_embedding_transform(q.x, coef0);
        k.x = rotary_embedding_transform(k.x, coef0);
        const auto coef1 = rotary_embedding_coefficient(4 * tid + 2, rot_embed_dim, base, scale, t_step);
        q.y = rotary_embedding_transform(q.y, coef1);
        k.y = rotary_embedding_transform(k.y, coef1);
    }

    inline __device__ void apply_rotary_embedding(uint4 &q, int tid, int rot_embed_dim, float base, float scale, int t_step)
    {
        if (8 * tid >= rot_embed_dim)
        {
            return;
        }
        const auto coef0 = rotary_embedding_coefficient(8 * tid, rot_embed_dim, base, scale, t_step);
        q.x = rotary_embedding_transform(q.x, coef0);
        const auto coef1 = rotary_embedding_coefficient(8 * tid + 2, rot_embed_dim, base, scale, t_step);
        q.y = rotary_embedding_transform(q.y, coef1);
        const auto coef2 = rotary_embedding_coefficient(8 * tid + 4, rot_embed_dim, base, scale, t_step);
        q.z = rotary_embedding_transform(q.z, coef2);
        const auto coef3 = rotary_embedding_coefficient(8 * tid + 6, rot_embed_dim, base, scale, t_step);
        q.w = rotary_embedding_transform(q.w, coef3);
    }

    inline __device__ void apply_rotary_embedding(
        uint4 &q, uint4 &k, int tid, int rot_embed_dim, float base, float scale, int t_step)
    {
        if (8 * tid >= rot_embed_dim)
        {
            return;
        }
        const auto coef0 = rotary_embedding_coefficient(8 * tid, rot_embed_dim, base, scale, t_step);
        q.x = rotary_embedding_transform(q.x, coef0);
        k.x = rotary_embedding_transform(k.x, coef0);
        const auto coef1 = rotary_embedding_coefficient(8 * tid + 2, rot_embed_dim, base, scale, t_step);
        q.y = rotary_embedding_transform(q.y, coef1);
        k.y = rotary_embedding_transform(k.y, coef1);
        const auto coef2 = rotary_embedding_coefficient(8 * tid + 4, rot_embed_dim, base, scale, t_step);
        q.z = rotary_embedding_transform(q.z, coef2);
        k.z = rotary_embedding_transform(k.z, coef2);
        const auto coef3 = rotary_embedding_coefficient(8 * tid + 6, rot_embed_dim, base, scale, t_step);
        q.w = rotary_embedding_transform(q.w, coef3);
        k.w = rotary_embedding_transform(k.w, coef3);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ void convert_from_float(float *dst, float src)
    {
        *dst = src;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ void convert_from_float(uint16_t *dst, float src)
    {
        *dst = float_to_half(src);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ void convert_from_float(uint32_t *dst, float2 src)
    {
        *dst = float2_to_half2(src);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ void convert_from_float(uint2 *dst, Float4_ src)
    {
        dst->x = float2_to_half2(src.x);
        dst->y = float2_to_half2(src.y);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ void convert_from_float(uint2 *dst, float4 src)
    {
        convert_from_float(dst, Float4_{make_float2(src.x, src.y), make_float2(src.z, src.w)});
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ void convert_from_float(uint4 *dst, Float8_ src)
    {
        dst->x = float2_to_half2(src.x);
        dst->y = float2_to_half2(src.y);
        dst->z = float2_to_half2(src.z);
        dst->w = float2_to_half2(src.w);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ void convert_from_float(float2 *dst, float2 src)
    {
        *dst = src;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ void convert_from_float(float4 *dst, float4 src)
    {
        *dst = src;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ void convert_from_float(Float8_ *dst, Float8_ src)
    {
        *dst = src;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <typename A>
    inline __device__ typename packed_type<float, num_elems<A>::value>::type convert_to_float(A u)
    {
        return {};
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <>
    inline __device__ float4 convert_to_float(float4 u)
    {
        return u;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <>
    inline __device__ float2 convert_to_float(float2 u)
    {
        return u;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <>
    inline __device__ float convert_to_float(float u)
    {
        return u;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <>
    inline __device__ Float8_ convert_to_float(uint4 u)
    {
        Float8_ f8;
        f8.x = half2_to_float2(u.x);
        f8.y = half2_to_float2(u.y);
        f8.z = half2_to_float2(u.z);
        f8.w = half2_to_float2(u.w);
        return f8;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <>
    inline __device__ float4 convert_to_float(uint2 u)
    {
        float4 ret;
        float2 f2x = half2_to_float2(u.x);
        float2 f2y = half2_to_float2(u.y);
        ret.x = f2x.x;
        ret.y = f2x.y;
        ret.z = f2y.x;
        ret.w = f2y.y;
        return ret;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <>
    inline __device__ float2 convert_to_float(uint32_t u)
    {
        return half2_to_float2(u);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <>
    inline __device__ float convert_to_float(half u)
    {
        return static_cast<float>(u);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ float float_from_int8(int8_t u)
    {
        return u;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ float2 float_from_int8(int16_t u)
    {
        union
        {
            int16_t int16;
            int8_t int8[2];
        };

        int16 = u;
        return make_float2(int8[0], int8[1]);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ float4 float_from_int8(int32_t u)
    {
        union
        {
            int32_t int32;
            int8_t int8[4];
        };

        int32 = u;
        return make_float4(int8[0], int8[1], int8[2], int8[3]);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    // clang-format off
inline __device__ Float8_ float_from_int8(int64_t u)
{
    union {
        int64_t int64;
        int16_t int16[4];
    };
    int64 = u;
    return Float8_ {float_from_int8(int16[0]),
                    float_from_int8(int16[1]),
                    float_from_int8(int16[2]),
                    float_from_int8(int16[3])};
}

    // clang-format on

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ float float_from_uint8(int8_t u)
    {
        union
        {
            int8_t int8;
            uint8_t uint8;
        };
        int8 = u;
        return uint8;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ float2 float_from_uint8(int16_t u)
    {
        union
        {
            int16_t int16;
            uint8_t uint8[2];
        };

        int16 = u;
        return make_float2(uint8[0], uint8[1]);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ float4 float_from_uint8(int32_t u)
    {
        union
        {
            int32_t int32;
            uint8_t uint8[4];
        };

        int32 = u;
        return make_float4(uint8[0], uint8[1], uint8[2], uint8[3]);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    // clang-format off
inline __device__ Float8_ float_from_uint8(int64_t u)
{
    union {
        int64_t int64;
        int16_t int16[4];
    };
    int64 = u;
    return Float8_ {float_from_uint8(int16[0]),
                    float_from_uint8(int16[1]),
                    float_from_uint8(int16[2]),
                    float_from_uint8(int16[3])};
}

    // clang-format on

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    // Start here
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ float2 float_from_uint4(int8_t u)
    {
        uint8_t u_1 = u & 0x0F;
        uint8_t u_2 = (u >> 4) & 0x0F;
        return make_float2(u_1, u_2);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ float4 float_from_uint4(int16_t u)
    {
        union
        {
            uint8_t uint8[4];
        };

        // int16 = u;
        uint8[0] = u & 0x000F;
        uint8[1] = (u >> 4) & 0x000F;
        uint8[2] = (u >> 8) & 0x000F;
        uint8[3] = (u >> 12) & 0x000F;
        return make_float4(uint8[0], uint8[1], uint8[2], uint8[3]);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    // clang-format off
inline __device__ Float8_ float_from_uint4(int32_t u)
{
    union {
        int32_t int32;
        int8_t int8[4];
    };
    int32 = u;
    return Float8_ {float_from_uint4(int8[0]),
                    float_from_uint4(int8[1]),
                    float_from_uint4(int8[2]),
                    float_from_uint4(int8[3])};
}

    // clang-format on

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    // Start here
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ float2 float_from_int4(int8_t u)
    {
        int8_t u_1 = u & 0x0F;
        int8_t u_2 = (u >> 4) & 0x0F;
        return make_float2(u_1, u_2);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ float4 float_from_int4(int16_t u)
    {
        union
        {
            int32_t int32;
            int8_t int8[4];
        };

        // int16 = u;
        int8[0] = u & 0x000F;
        int8[1] = (u >> 4) & 0x000F;
        int8[2] = (u >> 8) & 0x000F;
        int8[3] = (u >> 12) & 0x000F;
        return make_float4(int8[0], int8[1], int8[2], int8[3]);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    // clang-format off
inline __device__ Float8_ float_from_int4(int32_t u)
{
    union {
        int32_t int32;
        int8_t int8[4];
    };
    int32 = u;
    return Float8_ {float_from_int4(int8[0]),
                    float_from_int4(int8[1]),
                    float_from_int4(int8[2]),
                    float_from_int4(int8[3])};
}

    // clang-format on

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ int8_t cast_to_uint8(float val)
    {
        union
        {
            int8_t int8[2];
            int16_t int16;
        };

        asm volatile("cvt.rni.sat.u8.f32 %0, %1;" : "=h"(int16) : "f"(val));
        return int8[0];
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ int32_t cast_to_uint8(float2 val)
    {
        union
        {
            int8_t int8[2];
            int32_t int32;
        };

        int8[0] = cast_to_uint8(val.x);
        int8[1] = cast_to_uint8(val.y);
        return int32;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ int32_t cast_to_uint8(float4 val)
    {
        union
        {
            int8_t int8[4];
            int32_t int32;
        };

        int8[0] = cast_to_uint8(val.x);
        int8[1] = cast_to_uint8(val.y);
        int8[2] = cast_to_uint8(val.z);
        int8[3] = cast_to_uint8(val.w);
        return int32;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ int64_t cast_to_uint8(Float8_ val)
    {
        union
        {
            int8_t int8[8];
            int64_t int64;
        };

        int8[0] = cast_to_uint8(val.x.x);
        int8[1] = cast_to_uint8(val.x.y);
        int8[2] = cast_to_uint8(val.y.x);
        int8[3] = cast_to_uint8(val.y.y);
        int8[4] = cast_to_uint8(val.z.x);
        int8[5] = cast_to_uint8(val.z.y);
        int8[6] = cast_to_uint8(val.w.x);
        int8[7] = cast_to_uint8(val.w.y);
        return int64;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ int8_t cast_to_int8(float val)
    {
        union
        {
            int8_t int8[2];
            int16_t int16;
        };

        asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=h"(int16) : "f"(val));
        return int8[0];
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ int32_t cast_to_int8(float2 val)
    {
        union
        {
            int8_t int8[2];
            int32_t int32;
        };

        int8[0] = cast_to_int8(val.x);
        int8[1] = cast_to_int8(val.y);
        return int32;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ int32_t cast_to_int8(float4 val)
    {
        union
        {
            int8_t int8[4];
            int32_t int32;
        };

        int8[0] = cast_to_int8(val.x);
        int8[1] = cast_to_int8(val.y);
        int8[2] = cast_to_int8(val.z);
        int8[3] = cast_to_int8(val.w);
        return int32;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ int64_t cast_to_int8(Float8_ val)
    {
        union
        {
            int8_t int8[8];
            int64_t int64;
        };

        int8[0] = cast_to_int8(val.x.x);
        int8[1] = cast_to_int8(val.x.y);
        int8[2] = cast_to_int8(val.y.x);
        int8[3] = cast_to_int8(val.y.y);
        int8[4] = cast_to_int8(val.z.x);
        int8[5] = cast_to_int8(val.z.y);
        int8[6] = cast_to_int8(val.w.x);
        int8[7] = cast_to_int8(val.w.y);
        return int64;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    // One Single value cannot be cast into 2 int4
    // inline __device__ int8_t cast_to_packed_int4(float val)
    // {
    //     union
    //     {
    //         int8_t int8[2];
    //         int16_t int16;
    //     };

    //     asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=h"(int16) : "f"(val));
    //     return int8[0];
    // }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ int8_t cast_to_packed_uint4(float2 val)
    {
        union
        {
            int8_t int8[4];
            int16_t int16[2];
        };

        asm volatile("cvt.rni.sat.u8.f32 %0, %1;" : "=h"(int16[0]) : "f"(val.x));
        asm volatile("cvt.rni.sat.u8.f32 %0, %1;" : "=h"(int16[1]) : "f"(val.y));
        // original
        // int8[0] |= int8[2] << 4;
        int8[0] = (int8[0] & 0xF) | (int8[2] << 4);
        return int8[0];
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ int16_t cast_to_packed_uint4(float4 val)
    {
        union
        {
            int8_t int8[2];
            int16_t int16;
        };

        int8[0] = cast_to_packed_uint4(make_float2(val.x, val.y));
        int8[1] = cast_to_packed_uint4(make_float2(val.z, val.w));
        return int16;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ int32_t cast_to_packed_uint4(Float8_ val)
    {
        union
        {
            int8_t int8[4];
            int32_t int32;
        };

        int8[0] = cast_to_packed_uint4(make_float2(val.x.x, val.x.y));
        int8[1] = cast_to_packed_uint4(make_float2(val.y.x, val.y.y));
        int8[2] = cast_to_packed_uint4(make_float2(val.z.x, val.z.y));
        int8[3] = cast_to_packed_uint4(make_float2(val.w.x, val.w.y));
        return int32;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ int8_t cast_to_packed_int4(float2 val)
    {
        union
        {
            int8_t int8[4];
            int16_t int16[2];
        };

        asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=h"(int16[0]) : "f"(val.x));
        asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=h"(int16[1]) : "f"(val.y));
        // original
        // int8[0] |= int8[2] << 4;
        int8[0] = (int8[0] & 0xF) | (int8[2] << 4);
        return int8[0];
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ int16_t cast_to_packed_int4(float4 val)
    {
        union
        {
            int8_t int8[2];
            int16_t int16;
        };

        int8[0] = cast_to_packed_int4(make_float2(val.x, val.y));
        int8[1] = cast_to_packed_int4(make_float2(val.z, val.w));
        return int16;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ int32_t cast_to_packed_int4(Float8_ val)
    {
        union
        {
            int8_t int8[4];
            int32_t int32;
        };

        int8[0] = cast_to_packed_int4(make_float2(val.x.x, val.x.y));
        int8[1] = cast_to_packed_int4(make_float2(val.y.x, val.y.y));
        int8[2] = cast_to_packed_int4(make_float2(val.z.x, val.z.y));
        int8[3] = cast_to_packed_int4(make_float2(val.w.x, val.w.y));
        return int32;
    }

    template <typename V_vec_k>
    inline __device__ V_vec_k reorder_8xfp16(V_vec_k val){
        return val;
    }

    // special case, actually reorder to 04152637
    inline __device__ uint4 reorder_8xfp16(uint4 val)
    {
         uint4 ans = make_uint4(
            __byte_perm(val.x, val.z, 0x5410), 
            __byte_perm(val.x, val.z, 0x7632),
            __byte_perm(val.y, val.w, 0x5410), 
            __byte_perm(val.y, val.w, 0x7632)
         ); 
         // printf("%X,%X,%X,%X %X,%X,%X,%X\n", val.x, val.y, val.z, val.w, ans.x, ans.y, ans.z, ans.w);
         return ans;
    }

    template <typename V_vec_k>
    inline __device__ V_vec_k reorder_8xfp16_type2(V_vec_k val){
        return val;
    }

    // special case, actually reorder to 02461357
    inline __device__ uint4 reorder_8xfp16_type2(uint4 val)
    {
         uint4 ans = make_uint4(
            __byte_perm(val.x, val.y, 0x5410), 
            __byte_perm(val.z, val.w, 0x5410),
            __byte_perm(val.x, val.y, 0x7632), 
            __byte_perm(val.z, val.w, 0x7632)
         ); 
         // printf("%X,%X,%X,%X %X,%X,%X,%X\n", val.x, val.y, val.z, val.w, ans.x, ans.y, ans.z, ans.w);
         return ans;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <typename Vec_k, typename T, typename T_scale>
    inline __device__ void load_4bits_kv_cache_vec(Vec_k *vec, const T *pointer, int idx, T_scale scale)
    {
        ; // Not used.
    }

    template <typename Vec_k, typename T, typename T_scale>
    inline __device__ void load_8bits_kv_cache_vec(Vec_k *vec, const T *pointer, int idx, T_scale scale)
    {
        ; // Not used.
    }

    template <typename Vec_k, typename T, typename T_scale>
    inline __device__ void store_4bits_kv_cache_vec(T *pointer, const Vec_k &vec, int idx, T_scale scale)
    {
        ; // Not used.
    }
    template <typename Vec_k, typename T, typename T_scale>
    inline __device__ void store_8bits_kv_cache_vec(T *pointer, const Vec_k &vec, int idx, T_scale scale)
    {
        ; // Not used.
    }
    template <typename Vec_k, typename T, typename T_scale>
    inline __device__ void store_4bits_kv_cache_vec(T *pointer, const Vec_k &vec, int idx, T_scale scale, T_scale zero)
    {
        ; // Not used.
    }
    template <typename Vec_k, typename T, typename T_scale>
    inline __device__ void store_8bits_kv_cache_vec(T *pointer, const Vec_k &vec, int idx, T_scale scale, T_scale zero)
    {
        ; // Not used.
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <typename Vec_k>
    inline __device__ void load_8bits_kv_cache_vec(Vec_k *vec, const int8_t *pointer, int idx, float scale)
    {
        using Packed_8bits_t = typename packed_type<int8_t, num_elems<Vec_k>::value>::type;
        using Packed_Float_t = typename packed_type<float, num_elems<Vec_k>::value>::type;
        const auto quant = *reinterpret_cast<const Packed_8bits_t *>(&pointer[idx]);

        convert_from_float(vec, mul<Packed_Float_t>(scale, float_from_int8(quant)));
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <typename Vec_k>
    inline __device__ void load_4bits_kv_cache_vec(Vec_k *vec, const int8_t *pointer, int idx, float scale)
    {
        using Packed_8bits_t = typename packed_type<int8_t, num_elems<Vec_k>::value / 2>::type; // pack 2 4bits into an int8
        using Packed_Float_t = typename packed_type<float, num_elems<Vec_k>::value>::type;
        const auto quant = *reinterpret_cast<const Packed_8bits_t *>(&pointer[idx]);

        convert_from_float(vec, mul<Packed_Float_t>(scale, float_from_int4(quant)));
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <typename Vec_k>
    inline __device__ void store_8bits_kv_cache_vec(int8_t *pointer, const Vec_k &vec, int idx, float scale)
    {
        using Packed_8bits_t = typename packed_type<int8_t, num_elems<Vec_k>::value>::type;
        using Packed_Float_t = typename packed_type<float, num_elems<Vec_k>::value>::type;
        Packed_8bits_t out_quant = cast_to_int8(mul<Packed_Float_t>(scale, convert_to_float(vec)));

        *reinterpret_cast<Packed_8bits_t *>(&pointer[idx]) = out_quant;
    }

    template <typename Vec_k>
    inline __device__ void store_8bits_kv_cache_vec(int8_t *pointer, const Vec_k &vec, int idx, float scale, float zero)
    {
        using Packed_8bits_t = typename packed_type<int8_t, num_elems<Vec_k>::value>::type;
        using Packed_Float_t = typename packed_type<float, num_elems<Vec_k>::value>::type;
        Packed_8bits_t out_quant = cast_to_uint8(add(mul<Packed_Float_t>(scale, convert_to_float(vec)), zero));

        *reinterpret_cast<Packed_8bits_t *>(&pointer[idx]) = out_quant;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <typename Vec_k>
    inline __device__ void store_4bits_kv_cache_vec(int8_t *pointer, const Vec_k &vec, int idx, float scale)
    {
        using Packed_8bits_t = typename packed_type<int8_t, num_elems<Vec_k>::value / 2>::type; // pack 2 4bits into an int8
        using Packed_Float_t = typename packed_type<float, num_elems<Vec_k>::value>::type;
        Packed_8bits_t out_quant = cast_to_packed_int4(mul<Packed_Float_t>(scale, convert_to_float(vec)));

        *reinterpret_cast<Packed_8bits_t *>(&pointer[idx]) = out_quant;
    }

    template <typename Vec_k>
    inline __device__ void store_4bits_kv_cache_vec(int8_t *pointer, const Vec_k &vec, int idx, float scale, float zero)
    {
        using Packed_8bits_t = typename packed_type<int8_t, num_elems<Vec_k>::value / 2>::type; // pack 2 4bits into an int8
        using Packed_Float_t = typename packed_type<float, num_elems<Vec_k>::value>::type;
        Packed_8bits_t out_quant = cast_to_packed_uint4(add(mul<Packed_Float_t>(scale, convert_to_float(vec)), zero));

        *reinterpret_cast<Packed_8bits_t *>(&pointer[idx]) = out_quant;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    // TODO (Haotian): potential problem: if ENABLE_ZEROS, this should be float_from_uint8!!!
    template <typename Vec_in, typename Vec_out, typename T_cache, typename T_scale>
    inline __device__ void convert_from_8bit_kv_cache(Vec_out *vec_o, const Vec_in &vec_i, T_scale scale)
    {
        if constexpr (std::is_same<T_cache, int8_t>::value)
        {
            using Packed_Float_t = typename packed_type<float, num_elems<Vec_out>::value>::type;
            convert_from_float(vec_o, mul<Packed_Float_t>(scale, float_from_int8(vec_i)));
        }
        else
        {
            ; // not supported.
        }
    }

    template <typename Vec_in, typename Vec_out, typename T_cache, typename T_scale>
    inline __device__ void convert_from_8bit_kv_cache(Vec_out *vec_o, const Vec_in &vec_i, T_scale scale, T_scale zero)
    {
        if constexpr (std::is_same<T_cache, int8_t>::value)
        {
            using Packed_Float_t = typename packed_type<float, num_elems<Vec_out>::value>::type;
            convert_from_float(vec_o, mul<Packed_Float_t>(scale, sub(float_from_uint8(vec_i), zero)));
        }
        else
        {
            ; // not supported.
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <typename Vec_in, typename Vec_out, typename T_cache, typename T_scale>
    inline __device__ void convert_from_4bit_kv_cache(Vec_out *vec_o, const Vec_in &vec_i, T_scale scale)
    {
        if constexpr (std::is_same<T_cache, int8_t>::value)
        {
            using Packed_Float_t = typename packed_type<float, num_elems<Vec_out>::value>::type;
            convert_from_float(vec_o, mul<Packed_Float_t>(scale, float_from_int4(vec_i)));
        }
        else
        {
            ; // not supported.
        }
    }

    inline __device__ uint4 dequantize_s4_to_fp16x2(uint32_t const& source)
    {
        uint4 result;

        uint32_t*      h   = reinterpret_cast<uint32_t*>(&result);
        uint32_t const i4s = reinterpret_cast<uint32_t const&>(source);

        // First, we extract the i4s and construct an intermediate fp16 number.
        static constexpr uint32_t immLut                = (0xf0 & 0xcc) | 0xaa;
        static constexpr uint32_t BOTTOM_MASK           = 0x000f000f;
        static constexpr uint32_t TOP_MASK              = 0x00f000f0;
        static constexpr uint32_t I4s_TO_F16s_MAGIC_NUM = 0x64006400;

        // Note that the entire sequence only requires 1 shift instruction. This is thanks to the register packing
        // format and the fact that we force our integers to be unsigned, and account for this in the fp16 subtractions.
        // In addition, I exploit the fact that sub and fma have the same throughput in order to convert elt_23 and
        // elt_67 to fp16 without having to shift them to the bottom bits before hand.

        // Shift right by 8 to now consider elt_45 and elt_67. Issue first to hide RAW dependency if we issue
        // immediately before required.
        const uint32_t top_i4s = i4s >> 8;
        // Extract elt_01 - (i4s & 0x000f000f) | 0x64006400
        asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                        : "=r"(h[0])
                        : "r"(i4s), "n"(BOTTOM_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
        // Extract elt_23 (i4s & 0x00f000f0) | 0x64006400
        asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                        : "=r"(h[1])
                        : "r"(i4s), "n"(TOP_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
        // Extract elt_45 (top_i4s & 0x000f000f) | 0x64006400
        asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                        : "=r"(h[2])
                        : "r"(top_i4s), "n"(BOTTOM_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
        // Extract elt_67 (top_i4s & 0x00f000f0) | 0x64006400
        asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                        : "=r"(h[3])
                        : "r"(top_i4s), "n"(TOP_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));

        // I use inline PTX below because I am not sure if the compiler will emit float2half instructions if I use the
        // half2 ctor. In this case, I chose performance reliability over code readability.

        // This is the half2 {1032, 1032} represented as an integer.
        // static constexpr uint32_t FP16_TOP_MAGIC_NUM = 0x64086408;
        // Haotian: subtract {1024, 1024} instead, we do not need to map to [-8, 7]
        static constexpr uint32_t FP16_TOP_MAGIC_NUM = 0x64006400;
        // This is the half2 {1 / 16, 1 / 16} represented as an integer.
        static constexpr uint32_t ONE_SIXTEENTH = 0x2c002c00;
        // This is the half2 {-72, -72} represented as an integer.
        // static constexpr uint32_t NEG_72 = 0xd480d480;
        // Haotian: Let's use {-64, -64}.
        static constexpr uint32_t NEG_64 = 0xd400d400;

        // Finally, we construct the output numbers.
        // Convert elt_01
        asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[0]) : "r"(h[0]), "r"(FP16_TOP_MAGIC_NUM));
        // Convert elt_23
        asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(h[1]) : "r"(h[1]), "r"(ONE_SIXTEENTH), "r"(NEG_64));
        // Convert elt_45
        asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[2]) : "r"(h[2]), "r"(FP16_TOP_MAGIC_NUM));
        // Convert elt_67
        asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(h[3]) : "r"(h[3]), "r"(ONE_SIXTEENTH), "r"(NEG_64));

        return result;
    }

    template <typename Vec_in, typename Vec_out, typename T_cache, typename T_scale>
    inline __device__ void convert_from_4bit_kv_cache(Vec_out *vec_o, const Vec_in &vec_i, T_scale scale, T_scale zero)
    {
        // if constexpr (std::is_same<T_cache, int8_t>::value)
        // {
        //     using Packed_Float_t = typename packed_type<float, num_elems<Vec_out>::value>::type;
        //     convert_from_float(vec_o, mul<Packed_Float_t>(scale, sub(float_from_uint4(vec_i), zero)));
        // }
        // else
        // {
        //     ; // not supported.
        // }
        auto vec_o_half = reinterpret_cast<half2*>(vec_o);
        // // float x_before = __half2float(vec_o_half[0].x), y_before = __half2float(vec_o_half[0].y);

        half2 half_scale = make_half2(__float2half_rn(scale), __float2half_rn(scale));
        half2 half_zero = make_half2(__float2half_rn(-scale * zero), __float2half_rn(-scale * zero));
        *reinterpret_cast<uint4 *>(vec_o) = dequantize_s4_to_fp16x2(vec_i);
        #pragma unroll
        for (int i = 0; i < 4; i++){
            vec_o_half[i] = __hfma2(vec_o_half[i], half_scale, half_zero);
        }
        // printf("[before] %f %f, [after] %f %f\n", x_before, y_before, __half2float(vec_o_half[0].x), __half2float(vec_o_half[1].x));
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <typename T, bool INT8_KV_CACHE>
    struct kv_cache_type_t
    {
        using Type = T;
    };

    template <typename T>
    struct kv_cache_type_t<T, true>
    {
        using Type = int8_t;
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <typename T, typename T_cache>
    struct kv_cache_scale_type_t
    {
        using Type = float;
    };
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <typename Vec_T, typename T>
    __device__ __inline__ void vec_from_smem_transpose(Vec_T &vec, T *smem, int transpose_idx, int smem_pitch);

    template <>
    __device__ __inline__ void vec_from_smem_transpose(float &vec, float *smem, int transpose_idx, int smem_pitch)
    {
        return;
    }

    template <>
    __device__ __inline__ void vec_from_smem_transpose(uint32_t &vec, uint16_t *smem, int transpose_idx, int smem_pitch)
    {
        union
        {
            uint32_t u32;
            uint16_t u16[2];
        } tmp;

        tmp.u16[0] = smem[transpose_idx];
        tmp.u16[1] = smem[smem_pitch + transpose_idx];

        vec = tmp.u32;
    }

    template <>
    __device__ __inline__ void vec_from_smem_transpose(half2 &vec, half *smem, int transpose_idx, int smem_pitch)
    {
        return vec_from_smem_transpose(
            *reinterpret_cast<uint32_t *>(&vec), reinterpret_cast<uint16_t *>(smem), transpose_idx, smem_pitch);
    }

    template <>
    __device__ __inline__ void vec_from_smem_transpose(uint2 &vec, uint16_t *smem, int transpose_idx, int smem_pitch)
    {
        union
        {
            uint32_t u32;
            uint16_t u16[2];
        } tmp_1, tmp_2;

        tmp_1.u32 = *reinterpret_cast<uint32_t *>(&smem[transpose_idx]);
        tmp_2.u32 = *reinterpret_cast<uint32_t *>(&smem[smem_pitch + transpose_idx]);

        union
        {
            uint2 u32x2;
            uint16_t u16[4];
        } tmp_3;

        tmp_3.u16[0] = tmp_1.u16[0];
        tmp_3.u16[1] = tmp_2.u16[0];
        tmp_3.u16[2] = tmp_1.u16[1];
        tmp_3.u16[3] = tmp_2.u16[1];

        vec = tmp_3.u32x2;
    }

    template <>
    __device__ __inline__ void vec_from_smem_transpose(uint4 &vec, uint16_t *smem, int transpose_idx, int smem_pitch)
    {
        union
        {
            uint64_t u64;
            uint16_t u16[4];
        } tmp_1, tmp_2;

        tmp_1.u64 = *reinterpret_cast<uint64_t *>(&smem[transpose_idx]);
        tmp_2.u64 = *reinterpret_cast<uint64_t *>(&smem[smem_pitch + transpose_idx]);

        union
        {
            uint4 u32x4;
            uint16_t u16[8];
        } tmp_3;

        tmp_3.u16[0] = tmp_1.u16[0];
        tmp_3.u16[1] = tmp_2.u16[0];
        tmp_3.u16[2] = tmp_1.u16[1];
        tmp_3.u16[3] = tmp_2.u16[1];
        tmp_3.u16[4] = tmp_1.u16[2];
        tmp_3.u16[5] = tmp_2.u16[2];
        tmp_3.u16[6] = tmp_1.u16[3];
        tmp_3.u16[7] = tmp_2.u16[3];

        vec = tmp_3.u32x4;
    }

    template <>
    __device__ __inline__ void vec_from_smem_transpose(float4 &vec, float *smem, int transpose_idx, int smem_pitch)
    {
        vec.x = smem[transpose_idx];
        vec.z = smem[transpose_idx + 1];
        vec.y = smem[smem_pitch + transpose_idx];
        vec.w = smem[smem_pitch + transpose_idx + 1];
    }

    template <>
    __device__ __inline__ void vec_from_smem_transpose(uint32_t &vec, half *smem, int transpose_idx, int smem_pitch)
    {
        union
        {
            uint32_t u32;
            half u16[2];
        } tmp;

        tmp.u16[0] = smem[transpose_idx];
        tmp.u16[1] = smem[smem_pitch + transpose_idx];

        vec = tmp.u32;
    }

    template <>
    __device__ __inline__ void vec_from_smem_transpose(float2 &vec, float *smem, int transpose_idx, int smem_pitch)
    {
        vec.x = smem[transpose_idx];
        vec.y = smem[smem_pitch + transpose_idx];
    }

    template <typename Vec_T, typename T>
    __device__ __inline__ void write_smem_transpose(const Vec_T &vec, T *smem, int transpose_idx, int smem_pitch);

    template <>
    __device__ __inline__ void write_smem_transpose(const float &vec, float *smem, int transpose_idx, int smem_pitch)
    {
        return;
    }


    template <>
    __device__ __inline__ void write_smem_transpose(const uint4 &vec, uint16_t *smem, int transpose_idx, int smem_pitch)
    {
        union
        {
            uint64_t u64;
            uint16_t u16[4];
        } tmp_1, tmp_2;

        union
        {
            uint4 u32x4;
            uint16_t u16[8];
        } tmp_3;

        tmp_3.u32x4 = vec;
        tmp_1.u16[0] = tmp_3.u16[0];
        tmp_2.u16[0] = tmp_3.u16[1];
        tmp_1.u16[1] = tmp_3.u16[2];
        tmp_2.u16[1] = tmp_3.u16[3];
        tmp_1.u16[2] = tmp_3.u16[4];
        tmp_2.u16[2] = tmp_3.u16[5];
        tmp_1.u16[3] = tmp_3.u16[6];
        tmp_2.u16[3] = tmp_3.u16[7];

        *reinterpret_cast<uint64_t *>(&smem[transpose_idx]) = tmp_1.u64;
        *reinterpret_cast<uint64_t *>(&smem[smem_pitch + transpose_idx]) = tmp_2.u64;
    }

    template <>
    __device__ __inline__ void write_smem_transpose(const uint2 &vec, uint16_t *smem, int transpose_idx, int smem_pitch)
    {
        union
        {
            uint32_t u32;
            uint16_t u16[2];
        } tmp_1, tmp_2;

        union
        {
            uint2 u32x2;
            uint16_t u16[4];
        } tmp_3;

        tmp_3.u32x2 = vec;
        tmp_1.u16[0] = tmp_3.u16[0];
        tmp_2.u16[0] = tmp_3.u16[1];
        tmp_1.u16[1] = tmp_3.u16[2];
        tmp_2.u16[1] = tmp_3.u16[3];

        *reinterpret_cast<uint32_t *>(&smem[transpose_idx]) = tmp_1.u32;
        *reinterpret_cast<uint32_t *>(&smem[smem_pitch + transpose_idx]) = tmp_2.u32;
    }

    template <>
    __device__ __inline__ void write_smem_transpose(const uint32_t &vec, uint16_t *smem, int transpose_idx, int smem_pitch)
    {
        union
        {
            uint32_t u32;
            uint16_t u16[2];
        } tmp;

        tmp.u32 = vec;

        smem[transpose_idx] = tmp.u16[0];
        smem[smem_pitch + transpose_idx] = tmp.u16[1];
    }

    template <>
    __device__ __inline__ void write_smem_transpose(const float4 &vec, float *smem, int transpose_idx, int smem_pitch)
    {
        smem[transpose_idx] = vec.x;
        smem[transpose_idx + 1] = vec.z;
        smem[smem_pitch + transpose_idx] = vec.y;
        smem[smem_pitch + transpose_idx + 1] = vec.w;
    }

    template <>
    __device__ __inline__ void write_smem_transpose(const uint32_t &vec, half *smem, int transpose_idx, int smem_pitch)
    {
        union
        {
            uint32_t u32;
            half u16[2];
        } tmp;

        tmp.u32 = vec;
        smem[transpose_idx] = tmp.u16[0];
        smem[smem_pitch + transpose_idx] = tmp.u16[1];
    }

    template <>
    __device__ __inline__ void write_smem_transpose(const half2 &vec, half *smem, int transpose_idx, int smem_pitch)
    {
        return write_smem_transpose(*reinterpret_cast<const uint32_t *>(&vec), smem, transpose_idx, smem_pitch);
    }


    template <>
    __device__ __inline__ void write_smem_transpose(const float2 &vec, float *smem, int transpose_idx, int smem_pitch)
    {
        smem[transpose_idx] = vec.x;
        smem[smem_pitch + transpose_idx] = vec.y;
    }


    // For an explanation of next_power_of_two, see the following references:
    // https://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
    // https://stackoverflow.com/a/1322548
    template <typename T>
    __device__ __host__ std::enable_if_t<sizeof(T) == 1, T> constexpr next_power_of_two(T v)
    {
        --v;
        v |= v >> 1;
        v |= v >> 2;
        v |= v >> 4;
        return ++v;
    }

    template <typename T>
    __device__ __host__ std::enable_if_t<sizeof(T) == 2, T> constexpr next_power_of_two(T v)
    {
        --v;
        v |= v >> 1;
        v |= v >> 2;
        v |= v >> 4;
        v |= v >> 8;
        return ++v;
    }

    template <typename T>
    __device__ __host__ std::enable_if_t<sizeof(T) == 4, T> constexpr next_power_of_two(T v)
    {
        --v;
        v |= v >> 1;
        v |= v >> 2;
        v |= v >> 4;
        v |= v >> 8;
        v |= v >> 16;
        return ++v;
    }

    template <typename T>
    __device__ __host__ std::enable_if_t<sizeof(T) == 8, T> constexpr next_power_of_two(T v)
    {
        --v;
        v |= v >> 1;
        v |= v >> 2;
        v |= v >> 4;
        v |= v >> 8;
        v |= v >> 16;
        v |= v >> 32;
        return ++v;
    }

    template <typename T>
    __device__ __host__ constexpr inline T const &const_min(T const &a, T const &b)
    {
        return b < a ? b : a;
    }

    template <typename T>
    __device__ __host__ constexpr inline T const &const_max(T const &a, T const &b)
    {
        return b > a ? b : a;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ void apply_rotary_embedding(uint32_t &q, uint32_t q_pair, uint32_t &k, uint32_t k_pair, int tid0,
                                                  int tid1, int rot_embed_dim, float base, float scale, int t_step, int first_half)
    {
        const float2 coef0 = rotary_embedding_coefficient(tid0, rot_embed_dim, base, scale, t_step);
        const float2 coef1 = rotary_embedding_coefficient(tid1, rot_embed_dim, base, scale, t_step);
        float2 cos = make_float2(coef0.x, coef1.x);
        float2 sin = make_float2(coef0.y, coef1.y);
        float2 q_, k_;
        if (first_half)
        {
            q_ = sub(mul<float2>(cos, q), mul<float2>(sin, q_pair));
            k_ = sub(mul<float2>(cos, k), mul<float2>(sin, k_pair));
        }
        else
        {
            q_ = add(mul<float2>(cos, q), mul<float2>(sin, q_pair));
            k_ = add(mul<float2>(cos, k), mul<float2>(sin, k_pair));
        }
        // printf("[apply rope] %f %f %f %f\n", coef0.x, coef0.y, coef1.x, coef1.y);
        q = float2_to_half2(q_);
        k = float2_to_half2(k_);
    }

    template <typename Vec_type, typename Packed_type, typename T>
    inline __device__ void apply_rotary_embedding_gptneox(Vec_type &q, Vec_type &k, int tidx, int rotary_embedding_dim,
                                                          float rotary_embedding_base, float rotary_embedding_scale, int t_step, bool first_half)
    {
        // 32 threads: each hold VEC_SIZE elements (half)
        Vec_type q_pair, k_pair;
        constexpr int VEC_SIZE = sizeof(Vec_type) / sizeof(Packed_type);
        constexpr int PACKED_ELT_SIZE = sizeof(Packed_type) / sizeof(T);
        if constexpr (sizeof(Vec_type) == 2)
        {
            reinterpret_cast<uint16_t &>(q_pair) = __shfl_xor_sync(0xffffffff, reinterpret_cast<uint16_t &>(q), 16);
            reinterpret_cast<uint16_t &>(k_pair) = __shfl_xor_sync(0xffffffff, reinterpret_cast<uint16_t &>(k), 16);
        }
        else if constexpr (sizeof(Vec_type) == 4)
        {
            reinterpret_cast<unsigned int &>(q_pair) = __shfl_xor_sync(0xffffffff, reinterpret_cast<unsigned int &>(q), 16);
            reinterpret_cast<unsigned int &>(k_pair) = __shfl_xor_sync(0xffffffff, reinterpret_cast<unsigned int &>(k), 16);
        }
        else if constexpr (sizeof(Vec_type) >= 8)
        {
#pragma unroll
            for (int vec_id = 0; vec_id < sizeof(Vec_type) / 8; vec_id++)
            {
                reinterpret_cast<unsigned long *>(&q_pair)[vec_id] = __shfl_xor_sync(0xffffffff, reinterpret_cast<unsigned long *>(&q)[vec_id], 16);
                reinterpret_cast<unsigned long *>(&k_pair)[vec_id] = __shfl_xor_sync(0xffffffff, reinterpret_cast<unsigned long *>(&k)[vec_id], 16);
            }
        }

        const int half_rotary_dim = rotary_embedding_dim / 2;

#pragma unroll
        for (int elt_id = 0; elt_id < VEC_SIZE; elt_id++)
        {
            // Pack two elements for calculation (only one if each the thread only gets one element)
            // Assume the head size (or rotary embedding) is multiple of 8.
            const int rotary_emd_pos0_id = (tidx * VEC_SIZE * PACKED_ELT_SIZE + elt_id * PACKED_ELT_SIZE + 0 - int(!first_half) * half_rotary_dim) * 2;
            const int rotary_emd_pos1_id = (tidx * VEC_SIZE * PACKED_ELT_SIZE + elt_id * PACKED_ELT_SIZE + 1 - int(!first_half) * half_rotary_dim) * 2;

            const bool valid_rotary_pos = rotary_emd_pos1_id < rotary_embedding_dim;

            Packed_type q_ = reinterpret_cast<Packed_type *>(&q)[elt_id];
            Packed_type q_pair_ = reinterpret_cast<Packed_type *>(&q_pair)[elt_id];
            Packed_type k_ = reinterpret_cast<Packed_type *>(&k)[elt_id];
            Packed_type k_pair_ = reinterpret_cast<Packed_type *>(&k_pair)[elt_id];

            apply_rotary_embedding(q_, q_pair_, k_, k_pair_, rotary_emd_pos0_id, rotary_emd_pos1_id, rotary_embedding_dim,
                                   rotary_embedding_base, rotary_embedding_scale, t_step, first_half);

            if (valid_rotary_pos)
            {
                reinterpret_cast<Packed_type *>(&q)[elt_id] = q_;
                reinterpret_cast<Packed_type *>(&k)[elt_id] = k_;
            }
        }
    }

    template <typename T>
    inline __device__ float vec_max(T v);
    // float
    template <>
    inline __device__ float vec_max(float v)
    {
        return v;
    }

    template <>
    inline __device__ float vec_max(float2 v)
    {
        return fmaxf(fabsf(v.x), fabsf(v.y));
    }

    template <>
    inline __device__ float vec_max(float4 v)
    {
        return fmaxf(fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fabsf(v.z)), fabsf(v.w));
    }

    // half
    template <>
    inline __device__ float vec_max(uint v)
    {
        half2 v_h = (half2 &)v;
        return __half2float(__hmax(__habs(v_h.x), __habs(v_h.y)));
    }

    template <>
    inline __device__ float vec_max(uint2 v)
    {
        return fmaxf(vec_max(v.x), vec_max(v.y));
    }

    template <>
    inline __device__ float vec_max(uint4 v)
    {
        return fmaxf(fmaxf(fmaxf(vec_max(v.x), vec_max(v.y)), vec_max(v.z)), vec_max(v.w));
    }


    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <typename T>
    inline __device__ float vec_max_no_abs(T v);
    // float
    template <>
    inline __device__ float vec_max_no_abs(float v)
    {
        return v;
    }

    template <>
    inline __device__ float vec_max_no_abs(float2 v)
    {
        return fmaxf(v.x, v.y);
    }

    template <>
    inline __device__ float vec_max_no_abs(float4 v)
    {
        return fmaxf(fmaxf(fmaxf(v.x, v.y), v.z), v.w);
    }

    // half
    template <>
    inline __device__ float vec_max_no_abs(uint v)
    {
        half2 v_h = (half2 &)v;
        return __half2float(__hmax(v_h.x, v_h.y));
    }

    template <>
    inline __device__ float vec_max_no_abs(uint2 v)
    {
        return fmaxf(vec_max_no_abs(v.x), vec_max_no_abs(v.y));
    }

    template <>
    inline __device__ float vec_max_no_abs(uint4 v)
    {
        return fmaxf(fmaxf(fmaxf(vec_max_no_abs(v.x), vec_max_no_abs(v.y)), vec_max_no_abs(v.z)), vec_max_no_abs(v.w));
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <typename T>
    inline __device__ float vec_min_no_abs(T v);
    // float
    template <>
    inline __device__ float vec_min_no_abs(float v)
    {
        return v;
    }

    template <>
    inline __device__ float vec_min_no_abs(float2 v)
    {
        return fminf(v.x, v.y);
    }

    template <>
    inline __device__ float vec_min_no_abs(float4 v)
    {
        return fminf(fminf(fminf(v.x, v.y), v.z), v.w);
    }

    // half
    template <>
    inline __device__ float vec_min_no_abs(uint v)
    {
        half2 v_h = (half2 &)v;
        return __half2float(fminf(v_h.x, v_h.y));
    }

    template <>
    inline __device__ float vec_min_no_abs(uint2 v)
    {
        return fminf(vec_min_no_abs(v.x), vec_min_no_abs(v.y));
    }

    template <>
    inline __device__ float vec_min_no_abs(uint4 v)
    {
        return fminf(fminf(fminf(vec_min_no_abs(v.x), vec_min_no_abs(v.y)), vec_min_no_abs(v.z)), vec_min_no_abs(v.w));
    }

} // namespace mmha