# File authors: Haotian Tang, Shang Yang, Yujun Lin, Song Han
# @article{lin2024qserve,
#   title={QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving},
#   author={Lin*, Yujun and Tang*, Haotian and Yang*, Shang and Zhang, Zhekai and Xiao, Guangxuan and Gan, Chuang and Han, Song},
#   year={2024}
# }
import qserve_backend.qgemm_w4a8_per_chn
import qserve_backend.qgemm_w4a8_per_group
import torch


class W4A8OF16LinearDynamicInputScale(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        group_size: int = 128,
        device: torch.device = torch.cuda.current_device(),
    ):
        super().__init__()

        w_bit = 4
        self.interleave = 1  # Currently no interleave
        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.per_channel = group_size == -1
        self.group_size = group_size if group_size != -1 else in_features

        # quick sanity check (make sure aligment)
        assert self.in_features % self.group_size == 0
        assert out_features % (32 // self.w_bit) == 0

        int8_pack_num = 8 // self.w_bit

        assert out_features % (self.interleave) == 0
        self.register_buffer(
            "qweight",
            torch.zeros(
                (
                    out_features // self.interleave,
                    in_features // int8_pack_num * self.interleave,
                ),
                dtype=torch.int8,
                device=device,
            ).contiguous(),
        )

        self.register_buffer(
            "s1_scales",
            torch.zeros(
                (out_features,),
                dtype=torch.float16,
                device=device,
            ).contiguous(),
        )

        if self.per_channel:
            # per-channel quantization
            self.register_buffer(
                "s1_szeros",
                torch.zeros(
                    (out_features,),
                    dtype=torch.float16,
                    device=device,
                ).contiguous(),
                # NOTE: this is the scaled zeros
            )
            self.forward = self.forward_per_chn
        else:
            # per-group quantization
            self.register_buffer(
                "s2_scales",
                torch.zeros(
                    (
                        in_features // self.group_size,
                        out_features,
                    ),
                    dtype=torch.int8,
                    device=device,
                ).contiguous(),
            )
            self.register_buffer(
                "s2_zeros",
                torch.zeros(
                    (
                        in_features // self.group_size,
                        out_features,
                    ),
                    dtype=torch.int8,  # Actually 2's complement for sint8 zeros
                    device=device,
                ).contiguous(),
                # NOTE: this is the scaled zeros
            )
            self.forward = self.forward_per_group

        if bias:
            self.register_buffer(
                "bias", torch.zeros((out_features), dtype=torch.float16, device=device)
            )
        else:
            self.bias = None

    @torch.no_grad()
    def forward_per_chn(self, x, input_scales, input_sum, output_buffer):
        qserve_backend.qgemm_w4a8_per_chn.gemm_forward_cuda(
            x,
            self.qweight,
            self.s1_scales,
            input_scales,
            self.s1_szeros,
            input_sum,
            output_buffer,
        )
        output_bias = self.bias
        if output_bias is not None:
            output_buffer += output_bias

    @torch.no_grad()
    def forward_per_group(self, x, input_scales, input_sum, output_buffer):
        # input sum is of no use here. Only to keep the interface consistent
        qserve_backend.qgemm_w4a8_per_group.gemm_forward_cuda(
            x,
            self.qweight,
            self.s2_zeros,
            self.s2_scales,
            self.s1_scales,
            input_scales,
            output_buffer,
        )
        output_bias = self.bias
        if output_bias is not None:
            output_buffer += output_bias

    @classmethod
    def from_linear(
        cls,
        linear,
        w_bit,
        group_size,
        init_only=False,
        s1_scale=None,
        s2_scale=None,
        zeros=None,  # NOTE: zeros is NOT scaled, we scale it here. We only have zeros for the final stage quant.
    ):
        q_linear = cls(
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
            group_size,
            linear.weight.device,
        )
        if init_only:  # just prepare for loading sd
            return q_linear

        # need scales and zeros info for real quantization
        assert s1_scale is not None and zeros is not None
        if group_size != -1:  # per-group quantization
            assert s2_scale is not None

        if linear.bias is not None:
            q_linear.bias = linear.bias.clone().half()

        ## Quantize the weights
        if group_size != -1:  # per-group quantization
            # Step 1: Quantize the weights to int8
            linear_weight = linear.weight.data  # OC, IC
            linear_weight = linear_weight.div_(s1_scale.reshape(linear.out_features, 1).to(linear_weight.device))
            linear_weight = linear_weight.round_()
            # assert linear_weight.min() >= -119 and linear_weight.max() <= 119, "Stage 1: Quantized weight out of range" # 119 is the "magic" number
            assert (
                linear_weight.min() >= -128 and linear_weight.max() <= 127
            ), "Stage 1: Quantized weight out of range"

            # Step 2: Quantize the weights to int4
            linear_weight = linear_weight.reshape(
                linear.out_features, linear.in_features // group_size, group_size
            )
            s2_zero = zeros.reshape(
                linear.out_features, linear.in_features // group_size, 1
            )
            s2_scale = s2_scale.reshape(
                linear.out_features, linear.in_features // group_size, 1
            )
            linear_weight = linear_weight.div_(s2_scale.to(torch.float16).to(linear_weight.device)).add_(
                s2_zero.to(torch.float16).to(linear_weight.device)
            )
            assert (
                linear_weight.min() >= 0 and linear_weight.max() <= 15
            ), "Stage 2: Quantized weight out of range"

            # Step 3: Pack the fake quantized weights to real quantized weights
            # ---- Repack the weight ---- #
            # pack to M // 32, K // 32, (8, 4), ([2], 2, 2, 4)
            W_unpack_reorder = (
                linear_weight.reshape(
                    linear.out_features // 32,
                    2,
                    2,
                    8,
                    linear.in_features // 32,
                    2,
                    4,
                    4,
                )
                .permute(0, 4, 3, 6, 1, 5, 2, 7)
                .contiguous()
            )
            W_unpack_reorder = (
                W_unpack_reorder.permute(0, 1, 2, 3, 5, 6, 7, 4)
                .contiguous()
                .to(torch.int8)
            )
            # B_fp16_reorder = B_fp16_reorder[:, :, :, :, :, :, [3, 2, 1, 0]].contiguous()
            # [16, 0, 17, 1, ...]
            W_unpack_repacked = (W_unpack_reorder[..., 1] << 4) + W_unpack_reorder[
                ..., 0
            ]
            W_unpack_repacked = W_unpack_repacked.reshape(
                linear.out_features // 32, linear.in_features // 32, 32, 16
            )
            W_unpack_repacked = W_unpack_repacked.reshape(
                linear.out_features, linear.in_features // 2
            )
            q_linear.qweight.data[:, :] = W_unpack_repacked

            # ---- Pack the scales ---- #
            q_linear.s1_scales.data[:] = s1_scale.reshape(linear.out_features)

            s2_scale = (
                s2_scale.reshape(linear.out_features, linear.in_features // group_size)
                .transpose(0, 1)
                .contiguous()
            )
            s2_scale = s2_scale.reshape(
                linear.in_features // group_size, linear.out_features // 32, 32
            )
            s2_scale = (
                s2_scale.reshape(
                    linear.in_features // group_size, linear.out_features // 32, 4, 8
                )
                .transpose(-2, -1)
                .contiguous()
            )
            s2_scale = s2_scale.reshape(
                linear.in_features // group_size, linear.out_features
            ).contiguous()
            q_linear.s2_scales.data[:, :] = s2_scale

            # ---- Pack the zeros ---- #
            s2_zero = -s2_zero
            s2_zero = s2_zero.int()  # convert to 2-complement

            s2_zero = (
                s2_zero.reshape(linear.out_features, linear.in_features // group_size)
                .transpose(0, 1)
                .contiguous()
            )
            s2_zero = s2_zero.reshape(
                linear.in_features // group_size, linear.out_features // 32, 32
            )
            # for the last dimension, organize as 0, 8, 16, 24, 1, 9, 17, 25, ... following the requirement of tensor core gemm
            s2_zero = (
                s2_zero.reshape(
                    linear.in_features // group_size, linear.out_features // 32, 4, 8
                )
                .transpose(-2, -1)
                .contiguous()
            )
            s2_zero = (
                s2_zero.reshape(
                    linear.in_features // group_size, linear.out_features
                ).contiguous()
                * s2_scale
            )
            q_linear.s2_zeros.data[:, :] = s2_zero

        else:  # per-channel quantization
            # Step 1: Quantize the weights to int4
            linear_weight = linear.weight.data.cuda()  # OC, IC
            linear_weight = linear_weight.div_(s1_scale.reshape(linear.out_features, 1).to(linear_weight.device))
            linear_weight = linear_weight.round_().to(torch.int8)
            linear_weight = linear_weight.add_(zeros.reshape(linear.out_features, 1).to(linear_weight.device))

            assert (
                linear_weight.min() >= 0 and linear_weight.max() <= 15
            ), "Quantized weight out of range"

            # ---- Repack the weight ---- #
            # pack to M // 32, K // 32, (8, 4), ([2], 2, 2, 4)
            W_unpack_reorder = (
                linear_weight.reshape(
                    linear.out_features // 32,
                    2,
                    2,
                    8,
                    linear.in_features // 32,
                    2,
                    4,
                    4,
                )
                .permute(0, 4, 3, 6, 1, 5, 2, 7)
                .contiguous()
            )
            W_unpack_reorder = (
                W_unpack_reorder.permute(0, 1, 2, 3, 5, 6, 7, 4)
                .contiguous()
                .to(torch.int8)
            )
            # B_fp16_reorder = B_fp16_reorder[:, :, :, :, :, :, [3, 2, 1, 0]].contiguous()
            # [16, 0, 17, 1, ...]
            W_unpack_repacked = (W_unpack_reorder[..., 1] << 4) + W_unpack_reorder[
                ..., 0
            ]
            W_unpack_repacked = W_unpack_repacked.reshape(
                linear.out_features // 32, linear.in_features // 32, 32, 16
            )
            W_unpack_repacked = W_unpack_repacked.reshape(
                linear.out_features, linear.in_features // 2
            )
            q_linear.qweight.data[:, :] = W_unpack_repacked.contiguous()

            # ---- Pack the scales ---- #
            q_linear.s1_scales.data[:] = s1_scale.reshape(
                linear.out_features
            ).contiguous()
            q_linear.s1_szeros.data[:] = zeros.reshape(
                linear.out_features
            ).contiguous() * s1_scale.reshape(linear.out_features)

        return q_linear

    def extra_repr(self) -> str:
        return (
            "in_features={}, out_features={}, bias={}, w_bit={}, group_size={}".format(
                self.in_features,
                self.out_features,
                self.bias is not None,
                self.w_bit,
                self.group_size,
            )
        )
