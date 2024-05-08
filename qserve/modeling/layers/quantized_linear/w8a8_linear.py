# File authors: Haotian Tang, Shang Yang, Yujun Lin, Song Han
# @article{lin2024qserve,
#   title={QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving},
#   author={Lin*, Yujun and Tang*, Haotian and Yang*, Shang and Zhang, Zhekai and Xiao, Guangxuan and Gan, Chuang and Han, Song},
#   year={2024}
# }
from typing import Optional, Union

import qserve_backend.qgemm_w8a8 as qgemm
import torch


class W8A8OF16LinearStaticScale(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        scale: Union[torch.tensor, float] = 1.0,
        params_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        # Keep input parameters
        self.in_features = in_features
        self.out_features = out_features
        # size [1] or size [oc]
        self.register_buffer("dequant_scale", torch.ones(out_features))
        # Parameters.
        # NOTE: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        self.create_weights()

        if bias:
            self.bias = Parameter(
                torch.empty(
                    self.out_features,
                    device=torch.cuda.current_device(),
                    dtype=torch.float16,
                )
            )
        else:
            self.register_parameter("bias", None)

    def create_weights(self) -> None:
        self.register_buffer(
            "weight",
            torch.empty(
                self.out_features,
                self.in_features,
                dtype=torch.int8,
                requires_grad=False,
            ),
        )

    def apply_weights(
        self,
        x: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, input_):
        # Matrix multiply.
        output = self.apply_weights(input_, self.bias)
        output_bias = self.bias
        return output, output_bias


class W8A8OF16LinearDynamicInputScale(W8A8OF16LinearStaticScale):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        scale: Union[torch.tensor, float] = 1.0,
        params_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            scale=scale,
            params_dtype=params_dtype,
        )

    def apply_weights(
        self,
        # [batch, tokens, channels]
        x: torch.Tensor,
        # [batch * tokens]
        input_scale: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        x_shape = x.shape
        if len(x.shape) > 2:
            x = x.view(-1, x_shape[-1])
        y = qgemm.w8a8_gemm_forward_cuda(
            x, self.weight, self.dequant_scale.half(), input_scale.half()
        )
        if len(x.shape) > 2:
            y = y.view(*x_shape[:-1], -1)
        return y

    def forward(self, input_, input_scale):
        # Matrix multiply.
        output = self.apply_weights(input_, input_scale, self.bias)
        output_bias = self.bias
        return output, output_bias
