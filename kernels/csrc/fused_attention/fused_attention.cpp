// Adapted from NVIDIA/FasterTransformer and FlashAttention
// Modified by Haotian Tang and Shang Yang.
// @article{lin2024qserve,
//   title={QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving},
//   author={Lin*, Yujun and Tang*, Haotian and Yang*, Shang and Zhang, Zhekai and Xiao, Guangxuan and Gan, Chuang and Han, Song},
//   journal={arXiv preprint arXiv:2405.04532},
//   year={2024}
// }

#include <torch/extension.h>
#include "ATen/cuda/CUDAContext.h"
#include <c10/cuda/CUDAGuard.h>

#include "fused_attention.h"
#include "input_metadata_helper.h"
#include "update_kv_cache.h"
#include "decoderMaskedMultiheadAttention.h"
#include "kvCacheUtils.h"

#define CHECK_DEVICE(x) TORCH_CHECK(x.device().type() == torch::kCUDA, #x " must be on CUDA")
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

#define DISPATCH_FLOAT_AND_HALF_AND_BF16(TYPE, NAME, ...)                  \
  if (TYPE == at::ScalarType::Half) {                                      \
    using scalar_t = at::Half;                                             \
    __VA_ARGS__();                                                         \
  } else {                                                                 \
    AT_ERROR(#NAME, " not implemented for type '", toString(TYPE), "'"); \
  }


// template<typename T>
// void masked_multihead_attention(const Masked_multihead_attention_params<T>& params, const KVLinearBuffer& kv_cache_buffer,
//                                 const cudaStream_t& stream);

// template<typename T>
// void cross_multihead_attention(const tensorrt_llm::kernels::Masked_multihead_attention_params<T>& params, const tensorrt_llm::kernels::KVLinearBuffer& kv_cache_buffer,
//                                const cudaStream_t& stream);


template<typename T>
struct SATypeConverter {
    using Type = T;
};

template<>
struct SATypeConverter<at::Half> {
    using Type = uint16_t;
};

template<>
struct SATypeConverter<at::BFloat16> {
    using Type = __nv_bfloat16;
};

template <typename T>
void set_params(Masked_multihead_attention_params<T> &params,
                const size_t batch_size,
                const size_t nheads,
                const size_t nheads_kv,
                const size_t memory_max_seqlen,
                const size_t headdim,
                const int timestep,
                const int rotary_embedding_dim,
                const float rotary_base,
                const bool neox_rotary_style,
                const int qkv_batch_stride,
                T *q_ptr,
                T *k_ptr,
                T *v_ptr,
                // half* k_scale_orig_quant,
                // half* v_scale_orig_quant,
                // T *k_cache_ptr,
                // T *v_cache_ptr,
                int *length_per_sample,
                bool int4_kv_cache,
                bool kv_cache_with_zeros,
                float *alibi_slopes_ptr,
                T *out_ptr) {
    // Reset the parameters
    // memset(&params, 0, sizeof(params));
    params.q = q_ptr;
    params.k = k_ptr;
    params.v = v_ptr;
    params.q_bias = nullptr;
    params.k_bias = nullptr;
    params.v_bias = nullptr;
    // params.k_cache = k_cache_ptr;
    // params.v_cache = v_cache_ptr;
    // params.linear_bias_slopes = alibi_slopes_ptr;
    params.out = out_ptr;
    params.cache_indir = nullptr;
    // Haotian: be very careful about qkv_batch_stride.
    // k and v are not contiguous!!
    params.stride = qkv_batch_stride;
    params.batch_size = batch_size;
    params.beam_width = 1;
    params.memory_max_len = memory_max_seqlen;
    params.num_heads = nheads;
    params.num_kv_heads = nheads_kv;
    params.hidden_size_per_head = headdim;
    params.rotary_embedding_dim = rotary_embedding_dim;
    params.rotary_embedding_base = rotary_base;
    // params.k_scale_quant_orig = k_scale_quant_orig;
    // params.v_scale_quant_orig = v_scale_quant_orig;
    // params.k_scale_orig_quant = k_scale_orig_quant;
    // params.v_scale_orig_quant = v_scale_orig_quant;
    // params.neox_rotary_style = neox_rotary_style;
    params.timestep = timestep;
    params.inv_sqrt_dh = 1.f / sqrt(float(headdim));
    // params.total_padding_tokens = nullptr;
    // params.masked_tokens = nullptr;
    // params.prefix_prompt_lengths = nullptr;
    // params.max_prefix_prompt_length = 0;
    params.relative_attention_bias = nullptr;
    params.relative_attention_bias_stride = 0;
    // params.cross_attention_out = nullptr;
    params.max_decoder_seq_len = 0;
    // params.is_return_cross_attentions = false;
    params.finished = nullptr;
    params.memory_length_per_sample = nullptr;
    params.length_per_sample = length_per_sample;
    params.int4_kv_cache = int4_kv_cache;
    params.kv_cache_with_zeros = kv_cache_with_zeros;
    // std::cout << params.batch_size << " " << params.memory_max_len << " " << params.num_heads << " " << params.hidden_size_per_head << " " << params.timestep << std::endl;
}


// output = fused_attention.single_query_attention(
//   query,
//   key,
//   value,
//   # block_tables
//   input_metadata.block_tables,
//   # params.lengths_per_sample
//   input_metadata.context_lens,
//   # params.memory_max_len,
//   input_metadata.max_context_len,
//   # block_size,
//   block_size,
//   # params.linear_bias_slopes
//   alibi_slopes,
//   # RoPE parameters below: we do not apply RoPE in this kernel.
//   0,
//   10000,
//   True,
// )

torch::Tensor single_query_attention(const torch::Tensor q,
                                     const torch::Tensor k,
                                     const torch::Tensor v,
                                     const torch::Tensor kv_pointers, // B x 2 x M
                                     c10::optional<const torch::Tensor> length_per_sample_,
                                     c10::optional<const torch::Tensor> alibi_slopes_,
                                    //  c10::optional<const torch::Tensor> k_scale_orig_quant,
                                    //  c10::optional<const torch::Tensor> v_scale_orig_quant,
                                     int memory_max_seqlen,
                                     int tokens_per_block,
                                     int size_per_token,
                                     const int timestep,
                                     const int rotary_embedding_dim,
                                     const float rotary_base,
                                     // neox_rotary_style = not interleaved
                                     const bool neox_rotary_style,
                                     const bool int4_kv_cache,
                                     const bool kv_cache_with_zeros) {
    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v); CHECK_DEVICE(kv_pointers);
    int batch_size = kv_pointers.size(0);
    int nheads = q.size(1);
    int nheads_kv = k.size(1);
    int headdim = k.size(-1);
    // int max_blocks = kv_pointers.size(-1)
    // CHECK_SHAPE(q, batch_size, nheads, headdim);
    // CHECK_SHAPE(k, batch_size, nheads_kv, headdim);
    // CHECK_SHAPE(v, batch_size, nheads_kv, headdim);
    // CHECK_SHAPE(kv_pointers, batch_size, 2, max_blocks);
    // TORCH_CHECK(q.stride(2) == 1 && q.stride(1) == headdim);
    TORCH_CHECK(k.stride(2) == 1 && k.stride(1) == headdim);
    TORCH_CHECK(v.stride(2) == 1 && v.stride(1) == headdim);
    // TORCH_CHECK(q.stride(0) == k.stride(0) && q.stride(0) == v.stride(0));
    CHECK_CONTIGUOUS(kv_pointers);
    // CHECK_CONTIGUOUS(k);

    if (length_per_sample_.has_value()) {
        auto length_per_sample = length_per_sample_.value();
        CHECK_DEVICE(length_per_sample);
        CHECK_SHAPE(length_per_sample, batch_size);
        CHECK_CONTIGUOUS(length_per_sample);
        TORCH_CHECK(length_per_sample.dtype() == torch::kInt32);
    }

    if (alibi_slopes_.has_value()) {
      auto alibi_slopes = alibi_slopes_.value();
      CHECK_DEVICE(alibi_slopes);
      CHECK_SHAPE(alibi_slopes, nheads);
      CHECK_CONTIGUOUS(alibi_slopes); 
      TORCH_CHECK(alibi_slopes.dtype() == torch::kFloat32);
    }

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard{(char)q.get_device()};

    torch::Tensor out = torch::empty_like(q);

    DISPATCH_FLOAT_AND_HALF_AND_BF16(q.scalar_type(), "single_query_attention", [&] {
        using DataType = typename SATypeConverter<scalar_t>::Type;
        Masked_multihead_attention_params<DataType> params;
        // KVLinearBuffer kv_buffer{batch_size, 1, memory_max_seqlen, nheads_kv * headdim * sizeof(kv_cache.scalar_type())};
        KVBlockArray kv_buffer(batch_size, kv_pointers.size(-1), tokens_per_block, size_per_token);
        kv_buffer.data = reinterpret_cast<int64_t*>(kv_pointers.data_ptr());
        params.int8_kv_cache = true; 
        set_params(params, batch_size, nheads, nheads_kv, memory_max_seqlen, headdim, 
                   timestep, rotary_embedding_dim, rotary_base, neox_rotary_style, q.stride(0),
                   reinterpret_cast<DataType*>(q.data_ptr()),
                   reinterpret_cast<DataType*>(k.data_ptr()),
                   reinterpret_cast<DataType*>(v.data_ptr()),
                  //  k_scale_quant_orig.has_value()
                  //     ? reinterpret_cast<half**>(k_scale_quant_orig.value().data_ptr<long>()) : nullptr,
                  //  v_scale_quant_orig.has_value()
                  //     ? reinterpret_cast<half**>(v_scale_quant_orig.value().data_ptr<long>()) : nullptr,
                  //  k_scale_orig_quant.has_value()
                  //     ? reinterpret_cast<half*>(k_scale_orig_quant.value().data_ptr<at::Half>()) : nullptr,
                  //  v_scale_orig_quant.has_value()
                  //     ? reinterpret_cast<half*>(v_scale_orig_quant.value().data_ptr<at::Half>()) : nullptr,
                   // reinterpret_cast<DataType*>(k_cache.data_ptr()),
                   // reinterpret_cast<DataType*>(v_cache.data_ptr()),
                   length_per_sample_.has_value()
                       ? length_per_sample_.value().data_ptr<int>() : nullptr,
                   int4_kv_cache,
                   kv_cache_with_zeros,
                   alibi_slopes_.has_value() 
                       ? alibi_slopes_.value().data_ptr<float>(): nullptr,
                   reinterpret_cast<DataType*>(out.data_ptr()));
        auto stream = at::cuda::getCurrentCUDAStream();
        masked_multihead_attention(params, kv_buffer, stream);
    });
    return out;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
    "single_query_attention",
    &single_query_attention,
    "single query attention kernel from trtllm");
  m.def(
    "apply_bias_rope_update_kv_cache",
    &apply_bias_rope_update_kv_cache,
    "(context stage) add bias, apply rope and update kv cache");
  m.def(
    "compute_padding_offsets",
    &compute_padding_offsets,
    "compute padding offsets");
}
