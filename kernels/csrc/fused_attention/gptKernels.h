// Inspired by TRT-LLM.
// Modified by Haotian Tang and Shang Yang.
// @article{lin2024qserve,
//   title={QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving},
//   author={Lin*, Yujun and Tang*, Haotian and Yang*, Shang and Zhang, Zhekai and Xiao, Guangxuan and Gan, Chuang and Han, Song},
//   journal={arXiv preprint arXiv:2405.04532},
//   year={2024}
// }
/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>


enum class AttentionMaskType
{
    // Mask the padded tokens.
    PADDING = 0,
    // Mask the padded tokens and all the tokens that come after in a sequence.
    CAUSAL = 1,
    // See ChatGLM mask.
    BIDIRECTIONAL = 2
};

enum class PositionEmbeddingType : int8_t
{
    kLEARNED_ABSOLUTE = 0,
    kROPE_GPTJ = 1,
    kROPE_GPT_NEOX = 2,
    // Workflow: (bmm1_output * scale_bmm1 + alibi).
    kALIBI = 3,
    // Workflow: (bmm1_output + alibi) * scale_bmm1.
    kALIBI_WITH_SCALE = 4,
    kRELATIVE = 5
};

enum class RotaryScalingType : int8_t
{
    kNONE = 0,
    kLINEAR = 1,
    kDYNAMIC = 2,
};
