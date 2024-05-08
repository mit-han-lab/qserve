// Implemented by Haotian Tang and Shang Yang.
// @article{lin2024qserve,
//   title={QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving},
//   author={Lin*, Yujun and Tang*, Haotian and Yang*, Shang and Zhang, Zhekai and Xiao, Guangxuan and Gan, Chuang and Han, Song},
//   year={2024}
// }
#ifndef __GEMM_CUDA_KERNEL_W4A8_PER_CHN_CUH__
#define __GEMM_CUDA_KERNEL_W4A8_PER_CHN_CUH__

#include <cuda_fp16.h>
#include <cuda_pipeline_primitives.h>

#define OP_M 16
#define OP_N 8
#define OP_K 32
#define INTRIN_M 16
#define INTRIN_N 16
#define INTRIN_K 32
#define WARP_SIZE 32
#define SMEM_PAD_A 0
#define SMEM_PAD_B 0
#define PACK_SIZE 16
#if (__CUDACC_VER_MAJOR__ >= 11) && (__CUDACC_VER_MINOR__ >= 4)
#define L2_CACHEHINT(size) ".L2::" #size "B"
#else
#define L2_CACHEHINT(size)
#endif

template <int CTA_M, int CTA_N, int CTA_K, int WARP_M, int WARP_N, int WARP_K,
          int STAGES, int G>
__global__ void dense_kernel0(int8_t *__restrict__ A, int8_t *__restrict__ B,
                              half2 *__restrict__ wscales, half *__restrict__ ascales,
                              half2 *__restrict__ w_szs, half *__restrict__ a_ssums,
                              half *__restrict__ C, int M, int64_t N, int64_t K);

#endif