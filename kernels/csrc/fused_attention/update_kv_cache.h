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
                                              const bool kv_cache_with_zeros);
