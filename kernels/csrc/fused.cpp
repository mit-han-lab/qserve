// Inspired by vLLM-SmoothQuant: https://github.com/vllm-project/vllm/pull/1112 and TensorRT-LLM.
// Modified by Shang Yang and Haotian Tang.
// @article{lin2024qserve,
//   title={QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving},
//   author={Lin*, Yujun and Tang*, Haotian and Yang*, Shang and Zhang, Zhekai and Xiao, Guangxuan and Gan, Chuang and Han, Song},
//   journal={arXiv preprint arXiv:2405.04532},
//   year={2024}
// }
#include <torch/extension.h>
#include <cuda_fp16.h>

void invoke_dequant_add_residual(
    torch::Tensor &out,      // [..., hidden_size]
    torch::Tensor &input,    // [..., hidden_size]
    torch::Tensor &residual, // [..., hidden_size]
    at::Half scale);

void invoke_dequant_add_residual(
    torch::Tensor &out,      // [..., hidden_size]
    torch::Tensor &input,    // [..., hidden_size]
    torch::Tensor &residual, // [..., hidden_size]
    torch::Tensor &scale);   // [num_tokens]

void invoke_dequant(torch::Tensor &out,   // [..., hidden_size]
                    torch::Tensor &input, // [..., hidden_size]
                    at::Half scale);

void invoke_quant(torch::Tensor &out,   // [..., hidden_size]
                  torch::Tensor &input, // [..., hidden_size]
                  at::Half scale);

void invoke_quant(torch::Tensor &out,   // [..., hidden_size]
                  torch::Tensor &input, // [..., hidden_size]
                  torch::Tensor &scale);  // [num_tokens]

void invoke_quant_fuse_sum(torch::Tensor &out,   // [..., hidden_size]
                            torch::Tensor &input, // [..., hidden_size]
                            at::Half input_sum,
                            at::Half scale);

void invoke_quant_fuse_sum(torch::Tensor &out,   // [..., hidden_size]
                  torch::Tensor &input, // [..., hidden_size]
                  torch::Tensor &input_sum,
                  torch::Tensor &scale);  // [num_tokens]


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("invoke_dequant_add_residual",
        py::overload_cast<torch::Tensor &, torch::Tensor &, torch::Tensor &,
                          at::Half>(&invoke_dequant_add_residual),
        "Add the dequanted result and residual.");
  m.def("invoke_dequant_add_residual",
        py::overload_cast<torch::Tensor &, torch::Tensor &, torch::Tensor &,
                          torch::Tensor &>(&invoke_dequant_add_residual),
        "Add the dequanted result and residual.");
  m.def("invoke_dequant", &invoke_dequant, "Dequant.");
  m.def(
      "invoke_quant",
      py::overload_cast<torch::Tensor &, torch::Tensor &, at::Half>(&invoke_quant),
      "Quant.");
  m.def("invoke_quant", py::overload_cast<torch::Tensor &, torch::Tensor &, torch::Tensor &>(
      &invoke_quant),
      "Quant.");
  m.def(
      "invoke_quant_fuse_sum",
      py::overload_cast<torch::Tensor &, torch::Tensor &, at::Half, at::Half>(&invoke_quant_fuse_sum),
      "Quant & get input sum.");
  m.def("invoke_quant_fuse_sum", py::overload_cast<torch::Tensor &, torch::Tensor &, torch::Tensor &, torch::Tensor &>(
      &invoke_quant_fuse_sum),
      "Quant & get input sum.");
}
