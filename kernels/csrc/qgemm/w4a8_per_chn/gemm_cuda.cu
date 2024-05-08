// Implemented by Haotian Tang and Shang Yang.
// @article{lin2024qserve,
//   title={QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving},
//   author={Lin*, Yujun and Tang*, Haotian and Yang*, Shang and Zhang, Zhekai and Xiao, Guangxuan and Gan, Chuang and Han, Song},
//   year={2024}
// }
#include "gemm_cuda.h"
#include "gemm_cuda_kernel.cuh"
#include <cuda_fp16.h>
#include <cuda_pipeline_primitives.h>
#include <torch/extension.h>

// The kernel assumes that the tblock tile is at least as large as CTA_SIZE *
// PACK_SIZE. kernel only works for CTA_M/N=2*WARP_M/N?

// TODO: if we have correctness issue, check swizzle first

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
#define KERNEL_LAUNCH_CODE                                                                                   \
  constexpr int NUM_WARPS = (CTA_M / WARP_M) * (CTA_N / WARP_N) * (CTA_K / WARP_K);                          \
  constexpr int SCALES_SMEM_SIZE = (G >= CTA_K) ? (CTA_N * STAGES * 2) : (CTA_N * (CTA_K / G) * STAGES * 2); \
  constexpr int kSmemByteSize =                                                                              \
      ((CTA_M * (CTA_K + SMEM_PAD_A) + CTA_N * (CTA_K + SMEM_PAD_B) / 2) * STAGES + SCALES_SMEM_SIZE) *      \
      sizeof(int8_t);                                                                                        \
  if (kSmemByteSize >= 99 * 1024)                                                                            \
  {                                                                                                          \
    printf("This kernel requires %d Bytes of shared memory, which exceeds "                                  \
           "device limit.\n",                                                                                \
           kSmemByteSize);                                                                                   \
    return ;                                                                                       \
  }                                                                                                          \
  int num_blocks_m = (num_out_feats + CTA_M - 1) / CTA_M;                                                    \
  int num_blocks_n = num_out_channels / CTA_N / 1;                                                           \
  const int log_tile = get_log_tile<8>((num_out_feats + CTA_M - 1) / CTA_M);                                 \
  const int tile_shift = 1 << log_tile;                                                                      \
  dim3 num_blocks(num_blocks_n *tile_shift,                                                                  \
                  (num_blocks_m + tile_shift - 1) / tile_shift);                                             \
  dim3 threads_per_block(WARP_SIZE, NUM_WARPS);                                                              \
  auto kernel_func =                                                                                         \
      dense_kernel0<CTA_M, CTA_N, CTA_K, WARP_M, WARP_N, WARP_K, STAGES, G>;                                 \
  cudaFuncSetAttribute(kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize,                             \
                       kSmemByteSize);                                                                       \
  kernel_func<<<num_blocks, threads_per_block, kSmemByteSize>>>(                                             \
      in_feats, kernel, wscales, ascales, w_szs, a_ssums, out_feats, num_in_feats, num_out_channels,       \
       num_in_channels);

// n is the number of tiles on n dimension (# tokens)
template <int N>
__inline__ __host__ __device__ int get_log_tile(int n)
{
  if (N >= 8 && n >= 6)
    return 3;
  else if (N >= 4 && n >= 3)
    return 2;
  else if (N >= 2 && n >= 2)
    return 1;
  else
    return 0;
}

void gemm_forward_cuda(torch::Tensor _in_feats,
                        torch::Tensor _kernel,
                        torch::Tensor _wscales,
                        torch::Tensor _ascales,
                        torch::Tensor _w_szs,
                        torch::Tensor _a_ssums,
                        torch::Tensor _out_feats)
{
  int num_in_feats = _in_feats.size(0);
  int num_in_channels = _in_feats.size(1);
  // int kernel_volume = _out_in_map.size(1);
  auto in_feats = reinterpret_cast<int8_t *>(_in_feats.data_ptr<int8_t>());
  auto kernel = reinterpret_cast<int8_t *>(_kernel.data_ptr<int8_t>());
  auto w_szs = reinterpret_cast<half2 *>(_w_szs.data_ptr());
  auto a_ssums = reinterpret_cast<half *>(_a_ssums.data_ptr());
  auto wscales = reinterpret_cast<half2 *>(_wscales.data_ptr());
  auto ascales = reinterpret_cast<half *>(_ascales.data_ptr());
  // auto out_in_map = _out_in_map.data_ptr<int>();
  // auto options =
  //     torch::TensorOptions().dtype(torch::kHalf).device(_in_feats.device());
  // at::Tensor _out_feats =
  //     torch::empty({num_in_feats, _kernel.size(0)}, options);
  int num_out_feats = _out_feats.size(-2);
  int num_out_channels = _out_feats.size(-1);
  auto out_feats = reinterpret_cast<half *>(_out_feats.data_ptr<at::Half>());

  // blockIdx.x: i_factors[0] * j_factors[0]
  // blockIdx.y: i_factors[1] * j_factors[1]

  constexpr int G = 128;

  // very simple case. only consider M=128 and M=8192
  if (num_out_feats > 256)
  {
    constexpr int CTA_M = 128;
    constexpr int CTA_N = 128;
    constexpr int CTA_K = 64;
    constexpr int WARP_M = 64;
    constexpr int WARP_N = 32;
    constexpr int WARP_K = 64;
    constexpr int STAGES = 3;
    KERNEL_LAUNCH_CODE
  }
  else if (num_out_feats >= 128)
  {
    // for smaller workloads
    constexpr int CTA_M = 64;
    constexpr int CTA_N = 64;
    constexpr int CTA_K = 64;
    constexpr int WARP_M = 32;
    constexpr int WARP_N = 32;
    constexpr int WARP_K = 64;
    constexpr int STAGES = 4; // 6;
    KERNEL_LAUNCH_CODE
  }
  else
  {
    // for smaller workload;
    constexpr int CTA_M = 32;
    constexpr int CTA_N = 64;
    constexpr int CTA_K = 128;
    constexpr int WARP_M = 32;
    constexpr int WARP_N = 32;
    constexpr int WARP_K = 64;
    constexpr int STAGES = 3; // 6;
    KERNEL_LAUNCH_CODE
  }
  return ;
}
