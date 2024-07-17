// Inspired by TRT-LLM.
// Modified by Haotian Tang and Shang Yang.
// @article{lin2024qserve,
//   title={QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving},
//   author={Lin*, Yujun and Tang*, Haotian and Yang*, Shang and Zhang, Zhekai and Xiao, Guangxuan and Gan, Chuang and Han, Song},
//   journal={arXiv preprint arXiv:2405.04532},
//   year={2024}
// }
#pragma once
#include <torch/extension.h>


torch::Tensor single_query_attention(const torch::Tensor q,
                                     const torch::Tensor k,
                                     const torch::Tensor v,
                                     torch::Tensor kv_pointers, // B x 2 x M
                                     c10::optional<const torch::Tensor> length_per_sample_,
                                     c10::optional<const torch::Tensor> alibi_slopes_,
                                     int memory_max_seqlen,
                                     int tokens_per_block,
                                     int size_per_token,
                                     const int timestep,
                                     const int rotary_embedding_dim,
                                     const float rotary_base,
                                     // neox_rotary_style = not interleaved
                                     const bool neox_rotary_style,
                                     const bool int4_kv_cache,
                                     const bool kv_cache_with_zeros);