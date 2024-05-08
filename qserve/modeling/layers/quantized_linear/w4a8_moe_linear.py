import torch

# import my_moegemm as mygemm
# MoE release will come in the future.


class MoEW4A8OF16LinearDynamicInputScale(torch.nn.Module):
    def __init__(
        self,
        w_bit: int,
        num_experts: int,
        in_features: int,
        out_features: int,
        bias: bool = True,
        group_size: int = 128,
        device: torch.device = torch.cuda.current_device(),
    ):
        super().__init__()

        w_bit = 4
        self.interleave = 1  # Currently no interleave
        self.num_experts = num_experts
        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
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
                    num_experts,
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
                (
                    num_experts,
                    out_features,
                ),
                dtype=torch.float16,
                device=device,
            ).contiguous(),
        )
        self.register_buffer(
            "s1_szeros",
            torch.zeros(
                (
                    num_experts,
                    out_features,
                ),
                dtype=torch.float16,
                device=device,
            ).contiguous(),
            # NOTE: this is the scaled zeros
        )

        if bias:
            self.register_buffer(
                "bias",
                torch.zeros(
                    (num_experts, out_features), dtype=torch.float16, device=device
                ),
            )
        else:
            self.bias = None

    @torch.no_grad()
    def forward(self, x, input_scales, input_sum, problem_sizes):
        raise NotImplementedError("MOE implementation is yet to be released.")
        # output_buffer = mygemm.moe_gemm_forward_cuda_api(
        #     x,
        #     self.qweight,
        #     self.s1_scales,
        #     input_scales,
        #     self.s1_szeros,
        #     input_sum,
        #     problem_sizes,
        # )
        # return output_buffer

        # output_bias = self.bias
        # if output_bias is not None:
        #     output_buffer += output_bias

    @classmethod
    def from_linear(
        cls,
        linears,
        w_bit,
        group_size,
        init_only=False,
        s1_scales=None,
        s1_zeros=None,  # NOTE: s1_zero is NOT scaled, we scale it here
    ):
        q_linear = cls(
            w_bit,
            linears[0].in_features,
            linears[0].out_features,
            linears[0].bias is not None,
            group_size,
            linears[0].weight.device,
        )
        if s1_scales is not None:
            assert len(s1_scales) == len(linears)
        if s1_zeros is not None:
            assert len(s1_zeros) == len(linears)
        for expert_idx, linear in enumerate(linears):
            if init_only:  # just prepare for loading sd
                return q_linear

            s1_scale = s1_scales[expert_idx]
            s1_zero = s1_zeros[expert_idx]

            # need scales and zeros info for real quantization
            assert s1_scale is not None and s1_zero is not None

            if linear.bias is not None:
                q_linear.bias.data[expert_idx] = linear.bias.clone().half()

            ## Quantize the weights
            # Step 1: Quantize the weights to int4
            linear_weight = linear.weight.data  # OC, IC
            linear_weight = linear_weight.div_(s1_scale.reshape(linear.out_features, 1))
            linear_weight = linear_weight.round_().to(torch.int8)
            linear_weight = linear_weight.add_(s1_zero.reshape(linear.out_features, 1))

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
            q_linear.qweight.data[expert_idx, :, :] = W_unpack_repacked.contiguous()

            # ---- Pack the scales ---- #
            q_linear.s1_scales.data[expert_idx, :] = s1_scale.reshape(
                linear.out_features
            ).contiguous()
            q_linear.s1_szeros.data[expert_idx, :] = s1_zero.reshape(
                linear.out_features
            ).contiguous() * s1_scale.reshape(linear.out_features)

            # s2_scale = s2_scale.reshape(linear.out_features, linear.in_features // group_size).transpose(0, 1).contiguous()
            # s2_scale = s2_scale.reshape(linear.in_features // group_size, linear.out_features // 32, 32)
            # s2_scale = s2_scale.reshape(linear.in_features // group_size, linear.out_features // 32, 4, 8).transpose(-2, -1).contiguous()
            # s2_scale = s2_scale.reshape(linear.in_features // group_size, linear.out_features).contiguous()
            # q_linear.s2_scales.data[:,:] = s2_scale

            # # ---- Pack the zeros ---- #
            # s2_zero = - s2_zero
            # s2_zero = s2_zero.int()   # convert to 2-complement

            # s2_zero = s2_zero.reshape(linear.out_features, linear.in_features // group_size).transpose(0, 1).contiguous()
            # s2_zero = s2_zero.reshape(linear.in_features // group_size, linear.out_features // 32, 32)
            # # for the last dimension, organize as 0, 8, 16, 24, 1, 9, 17, 25, ... following the requirement of tensor core gemm
            # s2_zero = s2_zero.reshape(linear.in_features // group_size, linear.out_features // 32, 4, 8).transpose(-2, -1).contiguous()
            # s2_zero = s2_zero.reshape(linear.in_features // group_size, linear.out_features).contiguous() * s2_scale
            # q_linear.s2_zeros.data[:,:] = s2_zero

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
