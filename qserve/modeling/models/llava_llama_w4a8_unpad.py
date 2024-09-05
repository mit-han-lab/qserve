#    Modified from https://github.com/haotian-liu/LLaVA
#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from typing import Dict, List, Optional

import os
import warnings
import shutil
import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union
from qserve.utils.input_metadata import InputMetadata

from transformers import CLIPVisionModel
from transformers import LlamaConfig
from qserve.sampling_params import SamplingParams
from qserve.utils.input_metadata import InputMetadata
from qserve.utils.quant_config import QServeQuantConfig

from transformers.modeling_outputs import CausalLMOutputWithPast

from .llava_base.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from .llama_w4a8_unpad import LlamaForCausalLM, LlamaModel


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    def __init__(self, config, group_size, quant_kv_cache, kv_cache_config):
        super(LlavaLlamaModel, self).__init__(config, group_size, quant_kv_cache, kv_cache_config)

class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    def __init__(
        self,
        vlm_config,
        config: LlamaConfig,
        group_size: int,
        sampling_params: SamplingParams,
        quant_config: Optional[QServeQuantConfig] = QServeQuantConfig(weight_bits=4),
        kv_cache_config: Optional[Dict] = None,
        quant_path: Optional[str] = None,
    ) -> None:
        quant_kv_cache = True
        super(LlavaLlamaForCausalLM, self).__init__(vlm_config, group_size, sampling_params, quant_config, kv_cache_config, quant_path=None)
        self.model = LlavaLlamaModel(vlm_config, group_size, quant_kv_cache, kv_cache_config)
        # NOTE: Need to initialize Llava model first, then load the llm weights 
        self.load_weights(os.path.join(quant_path, "llm"))
        vision_tower = self.get_model().vision_tower
        # if not vision_tower.is_loaded:
        #     vision_tower.load_model()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        input_metadata: InputMetadata = None,
        start_pos: int = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        special_token: bool = False,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # if inputs_embeds is None:
        #     if special_token:
        #         (
        #             input_ids,
        #             position_ids,
        #             attention_mask,
        #             past_key_values,
        #             inputs_embeds,
        #             labels,
        #         ) = self.prepare_inputs_labels_for_multimodal(
        #             input_ids,
        #             position_ids,
        #             attention_mask,
        #             past_key_values,
        #             labels,
        #             images,
        #         )
        #     else:
        #         assert 0, "Wrong branch for input processing!"
        #         # inputs_embeds = self.default_inputs_embeds_for_multimodal(
        #         #     input_ids, inputs_embeds, images
        #         # )
        #         # input_ids = None

        # if start_pos == None:
        #     out = super().forward(
        #         input_ids=input_ids,
        #         attention_mask=attention_mask,
        #         position_ids=position_ids,
        #         past_key_values=past_key_values,
        #         inputs_embeds=inputs_embeds,
        #         labels=labels,
        #         use_cache=use_cache,
        #         output_attentions=output_attentions,
        #         output_hidden_states=output_hidden_states,
        #         return_dict=return_dict,
        #     )
        # else:
        out = super().forward(
            input_ids=input_ids,
            input_metadata=input_metadata,
        )
        return out

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs
    ):
        images = kwargs.pop("images", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        if images is not None:
            _inputs["images"] = images
        return _inputs
