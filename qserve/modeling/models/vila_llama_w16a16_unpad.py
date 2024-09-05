import os
import scipy
import numpy as np
import warnings
import shutil
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union

from qserve.sampling_params import SamplingParams
from qserve.utils.input_metadata import InputMetadata
from qserve.utils.quant_config import QServeQuantConfig

from transformers import AutoConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from llava.model.utils import get_model_config
from llava.model.language_model.builder import build_llm_and_tokenizer
from llava.model.multimodal_encoder.builder import build_vision_tower
from llava.model.multimodal_projector.builder import build_mm_projector
from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from .llama_w16a16_unpad import LlamaForCausalLM, LlamaModel

from llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
)

class VilaLlamaForCausalLM(LlavaMetaModel, LlavaMetaForCausalLM, PreTrainedModel):
    def __init__(self, 
                 config, 
                 sampling_params, 
                 quant_config: Optional[QServeQuantConfig] = QServeQuantConfig(weight_bits=4), 
                 kv_cache_config: Optional[Dict] = None,
                 quant_path: Optional[str] = None,
                 img_rotation: bool = False,
    ) -> None:
        super().__init__(config)
        self.img_rotation = img_rotation
        self.init_vlm(config, sampling_params, quant_config, kv_cache_config, quant_path)
    
    def init_vlm(self, config, sampling_params, quant_config, kv_cache_config, quant_path, *args, **kwargs):
        if hasattr(self, "llm") or hasattr(self, "vision_tower")  or hasattr(self, "mm_projector"):
            # already initialized, skipped
            return 
        
        model_dtype = getattr(config, "model_dtype", "torch.float16")
        if not hasattr(config, "model_dtype"):
            warnings.warn("model_dtype not found in config, defaulting to torch.float16.")
            config.model_dtype = model_dtype
        
        # print("init_vlm(): config", config); input("DEBUG init_vlm")
        cfgs = get_model_config(config)
        if len(cfgs) == 3:
            llm_cfg, vision_tower_cfg, mm_projector_cfg = cfgs
        else:
            raise ValueError("`llm_cfg` `mm_projector_cfg` `vision_tower_cfg` not found in the config.")
        # print("init_vlm():", cfgs); input("DEBUG init_vlm")
        llm_cfg = AutoConfig.from_pretrained(llm_cfg)
        
        # self.llm, self.tokenizer = build_llm_and_tokenizer(llm_cfg, config, *args, **kwargs)
        self.llm = LlamaForCausalLM(llm_cfg, sampling_params, quant_config, kv_cache_config, quant_path=None)
        self.vision_tower = build_vision_tower(vision_tower_cfg, config)
        self.mm_projector = build_mm_projector(mm_projector_cfg, config)

        self.post_config()
        # NOTE: Need to initialize Llava model first, then load the llm weights 
        if quant_path is not None:
            self.llm.load_weights(os.path.join(quant_path, "llm"))
        self.is_loaded = True

        assert (
            self.llm is not None or self.vision_tower is not None or self.mm_projector is not None
        ), "At least one of the components must be instantiated."
    
    def prepare_inputs_labels_for_multimodal(
        self, input_ids, past_key_values, labels, images
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if (
                past_key_values is not None
                and vision_tower is not None
                and images is not None
                and input_ids.shape[1] == 1
            ):
                target_shape = past_key_values[-1][-1].shape[-2] + 1
                # attention_mask = torch.cat(
                #     (
                #         attention_mask,
                #         torch.ones(
                #             (
                #                 attention_mask.shape[0],
                #                 target_shape - attention_mask.shape[1],
                #             ),
                #             dtype=attention_mask.dtype,
                #             device=attention_mask.device,
                #         ),
                #     ),
                #     dim=1,
                # )
                # position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
            return (
                input_ids,
                # position_ids,
                # attention_mask,
                past_key_values,
                None,
                labels,
            )
        # handle different image dtypes for packing
        if type(images) is list:
            images = torch.cat(images, dim=0)
        elif images.ndim == 5:  # batch_size x seq_len x image_channels
            images = images.flatten(0, 1)
        image_features = self.encode_images(images).to(self.device)
        
        # NOTE: Quick fix for rotation here.
        if self.img_rotation:
            raise ValueError("Image rotation is not supported in the current release.")
            size = 4096
            Q = scipy.linalg.hadamard(size) / np.sqrt(size)
            Q = torch.tensor(Q, dtype=torch.float64).to(image_features.device)
            imfeat_dtype = image_features.dtype
            image_features = torch.matmul(image_features.to(Q.dtype), Q).to(imfeat_dtype)

        # Note (kentang-mit@): image start / end is not implemented here to support pretraining.
        if getattr(self.config, "turn_mm_projector", False) and getattr(self.config, "mm_use_im_start_end", False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        # _position_ids = position_ids
        # _attention_mask = attention_mask
        # if attention_mask is None:
        #     attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        # else:
        #     attention_mask = attention_mask.bool()
        # if position_ids is None:
        #     position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask
        input_ids_copy = input_ids.clone()
        # kentang-mit@: Otherwise tokenizer out of bounds. Embeddings of image tokens will not be used.
        input_ids_copy[input_ids_copy == IMAGE_TOKEN_INDEX] = 0
        input_embeds = self.llm.model.embed_tokens(input_ids_copy)
        # print(self.llm.model.embed_tokens.weight)
        # print(self.llm.model.embed_tokens.weight.shape)
        # exit()

        # input_ids = [
        #     cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)
        # ]
        # input_embeds_1 = [
        #     cur_input_embeds[cur_attention_mask]
        #     for cur_input_embeds, cur_attention_mask in zip(input_embeds, attention_mask)
        # ]
        input_embeds_1 = input_embeds
        # labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0

        # print("BEFORE BATCH LOOP:", len(input_ids), input_ids[0].shape, input_ids[0].device, [(x == IMAGE_TOKEN_INDEX).sum() for x in input_ids])

        # kentang-mit@: If some part of the model is executed in the loop, the the loop length needs to be a constant.
        for batch_idx, cur_input_ids in enumerate(input_ids):
            cur_input_ids = input_ids[batch_idx]
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[0]
                # cur_input_embeds_1 = self.get_llm().embed_tokens(cur_input_ids)
                cur_input_embeds_1 = input_embeds_1[batch_idx]
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                # kenang-mit@: we do not have placeholdr image for text-only data now.
                # cur_image_idx += 1
                continue

            cur_input_embeds = input_embeds_1[batch_idx]
            image_token_indices = (
                [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            )
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            cur_input_embeds_no_im = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]])
                cur_input_embeds_no_im.append(cur_input_embeds[image_token_indices[i] + 1 : image_token_indices[i + 1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            # cur_input_embeds = self.get_llm().embed_tokens(torch.cat(cur_input_ids_noim))
            # cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []
            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(
                        torch.full(
                            (cur_image_features.shape[0],),
                            IGNORE_INDEX,
                            device=cur_labels.device,
                            dtype=cur_labels.dtype,
                        )
                    )

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # TODO: Remove this restriction due to seq batching, may need to consider later.
         
        # # Truncate sequences to max length as image embeddings can make the sequence longer
        # tokenizer_model_max_length = getattr(self.llm.config, "tokenizer_model_max_length", None)
        # if tokenizer_model_max_length is not None:
        #     if any(len(x) > tokenizer_model_max_length for x in new_input_embeds):
        #         warnings.warn("Inputs truncated!")
        #     new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
        #     new_labels = [x[:tokenizer_model_max_length] for x in new_labels]
        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full(
            (batch_size, max_len),
            IGNORE_INDEX,
            dtype=new_labels[0].dtype,
            device=new_labels[0].device,
        )
        # attention_mask = torch.zeros(
        #     (batch_size, max_len),
        #     dtype=attention_mask.dtype,
        #     device=attention_mask.device,
        # )
        # position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.llm.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                            cur_new_embed,
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    # attention_mask[i, -cur_len:] = True
                    # position_ids[i, -cur_len:] = torch.arange(
                    #     0, cur_len, dtype=position_ids.dtype, device=position_ids.device
                    # )
            else:
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            cur_new_embed,
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    # attention_mask[i, :cur_len] = True
                    # position_ids[i, :cur_len] = torch.arange(
                    #     0, cur_len, dtype=position_ids.dtype, device=position_ids.device
                    # )

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        # if _attention_mask is None:
        #     attention_mask = None
        # else:
        #     attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        # if _position_ids is None:
        #     position_ids = None

        return (
            None,
            None, #position_ids,
            None, #attention_mask,
            past_key_values,
            new_input_embeds,
            new_labels,
        )

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
        # self.freezed_module_patch()
        if inputs_embeds is None and input_metadata.is_prompt:
            (
                _,
                _,
                _,
                _,
                inputs_embeds,
                _,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids.unsqueeze(0), past_key_values, labels, input_metadata.image_tensor
            )
            inputs_embeds = inputs_embeds.squeeze(0)

        if inputs_embeds is not None:
            outputs = self.llm.forward(
                input_ids=None,
                input_metadata=input_metadata,
                inputs_embeds=inputs_embeds,
            )
        else:
            outputs = self.llm.forward(
                input_ids=input_ids,
                input_metadata=input_metadata,
            )
        return outputs