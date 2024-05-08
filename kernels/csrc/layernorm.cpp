// Inspired by TRT-LLM.
// Modified by Shang Yang and Haotian Tang.
// @article{lin2024qserve,
//   title={QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving},
//   author={Lin*, Yujun and Tang*, Haotian and Yang*, Shang and Zhang, Zhekai and Xiao, Guangxuan and Gan, Chuang and Han, Song},
//   year={2024}
// }
#include <torch/extension.h>
#include <cuda_fp16.h>

void rms_norm(torch::Tensor &out,    // [num_tokens, hidden_size]
              torch::Tensor &input,  // [num_tokens, hidden_size]
              torch::Tensor &weight, // [hidden_size]
              float epsilon, bool use_quant);

void rms_norm_general(torch::Tensor &out,    // [..., hidden_size]
              torch::Tensor &input,  // [..., hidden_size]
              torch::Tensor &weight, // [hidden_size]
              torch::Tensor &scaling, // [tokens] or [1]
              float epsilon,
              bool use_per_token_quant);

void rms_norm_general_fuse_sum(torch::Tensor &out,    // [..., hidden_size]
              torch::Tensor &input,  // [..., hidden_size]
              torch::Tensor &weight, // [hidden_size]
              torch::Tensor &input_sum, // [tokens] or [1]
              torch::Tensor &scaling, // [tokens] or [1]
              float epsilon,
              bool use_per_token_quant);

void invoke_dequant_add_residual_rms_norm_quant(
    torch::Tensor &out,      // [..., hidden_size]
    torch::Tensor &input,    // [..., hidden_size]
    torch::Tensor &residual, // [..., hidden_size]
    torch::Tensor &gamma,    // [hidden_size]
    at::Half scale, float epsilon);

void invoke_dequant_add_residual_rms_norm_quant(
    torch::Tensor &out,      // [..., hidden_size]
    torch::Tensor &input,    // [..., hidden_size]
    torch::Tensor &residual, // [..., hidden_size]
    torch::Tensor &gamma,    // [hidden_size]
    torch::Tensor &scale,    // [num_tokens]
    float epsilon);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rms_norm", &rms_norm, py::arg("out"), py::arg("input"),
        py::arg("weight"), py::arg("epsilon"), py::arg("use_quant") = false,
        "Apply Root Mean Square (RMS) Normalization to the input tensor.");

  m.def("rms_norm_general", &rms_norm_general, py::arg("out"), py::arg("input"),
        py::arg("weight"), py::arg("scaling"), py::arg("epsilon"), py::arg("use_per_token_quant") = false,
        "Apply Root Mean Square (RMS) Normalization to the input tensor (TRTLLM kernel).");

  m.def("rms_norm_general_fuse_sum", &rms_norm_general_fuse_sum, py::arg("out"), py::arg("input"),
        py::arg("weight"), py::arg("input_sum"), py::arg("scaling"), py::arg("epsilon"), py::arg("use_per_token_quant") = false,
        "Apply Root Mean Square (RMS) Normalization to the input tensor & get input sum (TRTLLM kernel).");

  m.def("invoke_dequant_add_residual_rms_norm_quant",
        py::overload_cast<torch::Tensor &, torch::Tensor &, torch::Tensor &,
                          torch::Tensor &, at::Half, float>(
            &invoke_dequant_add_residual_rms_norm_quant),
        "Add the dequanted result and residual, then use RMS norm and quant "
        "output.");
  m.def("invoke_dequant_add_residual_rms_norm_quant",
        py::overload_cast<torch::Tensor &, torch::Tensor &, torch::Tensor &,
                          torch::Tensor &, torch::Tensor &, float>(
            &invoke_dequant_add_residual_rms_norm_quant),
        "Add the dequanted result and residual, then use RMS norm and quant "
        "output.");
}
