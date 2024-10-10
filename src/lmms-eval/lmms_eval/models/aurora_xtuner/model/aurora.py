from collections import OrderedDict
from typing import Any, Optional, Tuple, Union
import warnings

import json
import os.path as osp
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from mmengine.config import Config, ConfigDict
from mmengine.model import BaseModel
from mmengine import print_log
from peft import get_peft_model, prepare_model_for_kbit_training
from einops import rearrange
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from transformers.models.clip.configuration_clip import CLIPConfig, CLIPVisionConfig
from transformers.models.siglip.configuration_siglip import SiglipConfig, SiglipTextConfig, SiglipVisionConfig
from transformers.models.clip.modeling_clip import CLIPVisionTransformer, CLIPMLP
from transformers.models.siglip.modeling_siglip import SiglipVisionTransformer, SiglipMLP

from transformers import (AutoModel, AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, CLIPImageProcessor,
                          CLIPVisionModel, GenerationConfig,
                          SiglipVisionModel)

from accelerate import init_empty_weights

from xtuner.registry import BUILDER
from xtuner.utils import IGNORE_INDEX, IMAGE_TOKEN_INDEX
from xtuner.model.modules import ProjectorConfig, ProjectorModel, dispatch_modules
from xtuner.model.tome import bipartite_soft_matching, merge_wavg
from xtuner.model.utils import (
    LoadWoInit,
    find_all_linear_names,
    get_peft_model_state_dict,
    guess_load_checkpoint,
    make_inputs_require_grad,
    prepare_inputs_labels_for_multimodal,
    prepare_inputs_labels_for_multimodal_slowfast,
    traverse_dict,
)

def convert_state_dict_to_hf(state_dict, mapping):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.endswith('.inv_freq'):
            continue
        for key_to_modify, new_key in mapping.items():
            if key_to_modify in key:
                key = key.replace(key_to_modify, new_key)
        new_state_dict[key] = value
    return new_state_dict

class AuroraModel(BaseModel):
    def __init__(
        self,
        llm,
        visual_encoder,
        freeze_llm=False,
        freeze_visual_encoder=False,
        freeze_proj=False,
        visual_select_layer=-2,
        pretrained_pth=None,
        projector_depth=2,
        llm_lora=None,
        visual_encoder_lora=None,
        use_activation_checkpointing=True,
        beta=0.1,
        label_smoothing=0.0,
        slowfast=False,
    ):
        super().__init__()
        self.freeze_llm = freeze_llm
        self.freeze_visual_encoder = freeze_visual_encoder
        self.freeze_proj = freeze_proj

        with LoadWoInit():
            self.llm = self._build_from_cfg_or_module(llm)
            self.visual_encoder = self._build_from_cfg_or_module(visual_encoder)


        self.llm.config.use_cache = False
        dispatch_modules(self.llm)

        projector_config = ProjectorConfig(
            visual_hidden_size=self.visual_encoder.config.hidden_size,
            llm_hidden_size=self.llm.config.hidden_size,
            depth=projector_depth,
        )
        self.projector = ProjectorModel(projector_config).to(self.visual_encoder.dtype)

        if self.freeze_llm:
            self.llm.requires_grad_(False)
            print("Freeze LLM")
        if self.freeze_visual_encoder:
            self.visual_encoder.requires_grad_(False)
            print("Freeze visual encoder")
        if self.freeze_proj:
            self.projector.requires_grad_(False)
            print("Freeze Proj")

        def check_requires_grad(model):
            for name, param in model.named_parameters():
                print(f"{name}: {param.requires_grad}")


        if use_activation_checkpointing:
            # For backward compatibility
            if hasattr(self.llm, "enable_input_require_grads"):
                self.llm.enable_input_require_grads()
            else:
                self.llm.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
            if hasattr(self.visual_encoder, "enable_input_require_grads"):
                self.visual_encoder.enable_input_require_grads()
            else:
                self.visual_encoder.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
            self.projector.enable_input_require_grads()

            # enable gradient (activation) checkpointing for memory efficiency
            self.gradient_checkpointing_enable()

        self.use_llm_lora = llm_lora is not None
        self.use_visual_encoder_lora = visual_encoder_lora is not None

        if self.use_llm_lora:
            self._prepare_llm_for_lora(llm_lora, use_activation_checkpointing)
        if self.use_visual_encoder_lora:
            self._prepare_visual_encoder_for_lora(visual_encoder_lora, use_activation_checkpointing)

        if pretrained_pth is not None:
            if 'projector' in pretrained_pth:
                with LoadWoInit():
                    self.projector = AutoModel.from_pretrained(
                        pretrained_pth)
            else:
                pretrained_state_dict = guess_load_checkpoint(pretrained_pth)
                self.load_state_dict(pretrained_state_dict, strict=False)
            print(f"Load pretrained weight from {pretrained_pth}")

        self.projector_depth = projector_depth
        self.visual_select_layer = visual_select_layer

        self._is_init = True
        
        self.slowfast = slowfast

    def _parse_lora_config(self, lora_config):
        if isinstance(lora_config, dict) or isinstance(lora_config, Config) or isinstance(lora_config, ConfigDict):
            lora_config = BUILDER.build(lora_config)
        return lora_config

    def _prepare_llm_for_lora(self, lora_config, use_activation_checkpointing=True):
        lora_config = self._parse_lora_config(lora_config)
        self.llm = prepare_model_for_kbit_training(self.llm, use_activation_checkpointing)
        if lora_config.target_modules is None:
            modules = find_all_linear_names(self.llm)
            lora_config.target_modules = modules
        self.llm = get_peft_model(self.llm, lora_config)

    def _prepare_visual_encoder_for_lora(self, lora_config, use_activation_checkpointing=True):
        lora_config = self._parse_lora_config(lora_config)
        if lora_config.target_modules is None:
            modules = find_all_linear_names(self.visual_encoder)
            lora_config.target_modules = modules
        self.visual_encoder = get_peft_model(self.visual_encoder, lora_config)

    def gradient_checkpointing_enable(self):
        self.activation_checkpointing_enable()

    def activation_checkpointing_enable(self):
        self.llm.gradient_checkpointing_enable()
        self.visual_encoder.gradient_checkpointing_enable()
        self.projector.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.activation_checkpointing_disable()

    def activation_checkpointing_disable(self):
        self.llm.gradient_checkpointing_disable()
        self.visual_encoder.gradient_checkpointing_disable()
        self.projector.gradient_checkpointing_disable()

    def init_weights(self):
        pass

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        to_return = OrderedDict()
        # Step 1. visual_encoder
        if self.use_visual_encoder_lora:
            to_return.update(get_peft_model_state_dict(self.visual_encoder, state_dict=state_dict))
        elif not self.freeze_visual_encoder:
            to_return.update({k: v for k, v in state_dict.items() if "visual_encoder." in k})
        # Step 2. LLM
        if self.use_llm_lora:
            to_return.update(get_peft_model_state_dict(self.llm, state_dict=state_dict))
        elif not self.freeze_llm:
            to_return.update({k: v for k, v in state_dict.items() if "llm." in k})
        # Step 3. Projector
        to_return.update({k: v for k, v in state_dict.items() if "projector." in k})
        return to_return

    def _build_from_cfg_or_module(self, cfg_or_mod):
        if isinstance(cfg_or_mod, nn.Module):
            return cfg_or_mod
        elif isinstance(cfg_or_mod, dict):
            traverse_dict(cfg_or_mod)
            return BUILDER.build(cfg_or_mod)
        else:
            raise NotImplementedError

    def forward(self, data, data_samples=None, mode="loss"): 
        if "pixel_values" in data:
            # make image a single frame video
            if data["pixel_values"].ndim == 4:
                data["pixel_values"] = data["pixel_values"].unsqueeze(1) 
            b, f = data["pixel_values"].shape[0], data["pixel_values"].shape[1]
            data["pixel_values"] = rearrange(data["pixel_values"], "b f c h w -> (b f) c h w")
            

            if self.slowfast and f != 1: # b = 1
                low_res_frame = data["pixel_values"][1:].to(self.visual_encoder.dtype)
                low_visual_outputs = self.visual_encoder(low_res_frame, output_hidden_states=True)
                low_visual_outputs = low_visual_outputs.hidden_states[self.visual_select_layer][:, 1:]
                low_visual_outputs = rearrange(low_visual_outputs, "(b f) n c -> b (f n) c", b=b)
                low_visual_outputs = self.projector(low_visual_outputs)
                low_visual_outputs = rearrange(low_visual_outputs, "b (f n) c -> b f n c", f=f-1)

                self.visual_encoder.visual_token_merge_ratio = 1.0
                high_res_frame = data["pixel_values"][:1].to(self.visual_encoder.dtype)
                high_visual_outputs = self.visual_encoder(high_res_frame, output_hidden_states=True)
                high_visual_outputs = high_visual_outputs.hidden_states[self.visual_select_layer][:, 1:]
                high_visual_outputs = rearrange(high_visual_outputs, "(b f) n c -> b (f n) c", b=b)
                high_visual_outputs = self.projector(high_visual_outputs)
                high_visual_outputs = rearrange(high_visual_outputs, "b (f n) c -> b f n c", f=1)

                visual_outputs = []
                visual_outputs.append(high_visual_outputs.squeeze(0))
                low_visual_outputs = low_visual_outputs.squeeze(0)
                for i in range(low_visual_outputs.size(0)):
                    visual_outputs.append(low_visual_outputs[i])

                data["pixel_values"] = visual_outputs
                data = prepare_inputs_labels_for_multimodal_slowfast(llm=self.llm, **data)
            else:
                try:
                    visual_outputs = self.visual_encoder(data["pixel_values"], output_hidden_states=True)
                except:
                    data["pixel_values"] = data["pixel_values"].to(self.visual_encoder.dtype)
                    visual_outputs = self.visual_encoder(data["pixel_values"], output_hidden_states=True)
                visual_outputs = visual_outputs.hidden_states[self.visual_select_layer][:, 1:]
                visual_outputs = rearrange(visual_outputs, "(b f) n c -> b (f n) c", b=b)
                visual_outputs = self.projector(visual_outputs)
                visual_outputs = rearrange(visual_outputs, "b (f n) c -> b f n c", f=f)
                data["pixel_values"] = visual_outputs
                data = prepare_inputs_labels_for_multimodal(llm=self.llm, **data)
            

        if mode == "loss":
            return self.compute_loss(data, data_samples)
        elif mode == "predict":
            return self.predict(data, data_samples)
        elif mode == "tensor":
            return self._forward(data, data_samples)
        elif mode == "inference":
            return data
        else:
            raise NotImplementedError

    def _forward(self, data, data_samples=None):

        outputs = self.llm(**data)

        return outputs

    def predict(self, data, data_samples=None):
        outputs = self.llm(**data)
        logits_dict = [{"logits": logits} for logits in outputs.logits]
        return logits_dict

    def compute_loss(self, data, data_samples=None):
        outputs = self.llm(**data)
        loss_dict = {"loss": outputs.loss}
        return loss_dict


    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.llm, name)

    def to_hf(self,
            cfg,
            save_dir,
            fp32=False,
            save_pretrained_kwargs={},
            save_format='xtuner',
            **kwargs):
        if save_format == 'xtuner':
            self.to_xtuner_llava(cfg, save_dir, fp32, save_pretrained_kwargs)
        elif save_format == 'huggingface':
            self.to_huggingface_llava(cfg, save_dir, fp32,
                                      save_pretrained_kwargs)
        elif save_format == 'official':
            self.to_official_llava(cfg, save_dir, fp32, save_pretrained_kwargs)
        else:
            raise NotImplementedError

    def to_xtuner_llava(self,
                        cfg,
                        save_dir,
                        fp32=False,
                        save_pretrained_kwargs={}):
        # LLM
        self.llm.config.use_cache = True
        # if not fp32:
        #     print_log('Convert LLM to float16', 'current')
        #     self.llm.half()
        if self.use_llm_lora:
            llm_path = osp.join(save_dir, 'llm_adapter')
            print_log(f'Saving LLM adapter to {llm_path}', 'current')
            self.llm.save_pretrained(llm_path, **save_pretrained_kwargs)
        elif not self.freeze_llm:
            llm_path = save_dir
            print_log(f'Saving LLM tokenizer to {llm_path}', 'current')
            tokenizer = BUILDER.build(cfg.tokenizer)
            tokenizer.save_pretrained(llm_path, **save_pretrained_kwargs)
            print_log(f'Saving LLM to {llm_path}', 'current')
            self.llm.generation_config.temperature=None
            self.llm.generation_config.top_p=None
            self.llm.save_pretrained(llm_path, **save_pretrained_kwargs)
        self.llm.config.use_cache = False

        # Visual Encoder
        if self.use_visual_encoder_lora:
            visual_encoder_path = osp.join(save_dir, 'visual_encoder_adapter')
            print_log(
                f'Saving visual_encoder adapter to {visual_encoder_path}',
                'current')
            self.visual_encoder.save_pretrained(visual_encoder_path,
                                                **save_pretrained_kwargs)
        elif not self.freeze_visual_encoder:
            visual_encoder_path = osp.join(save_dir, 'visual_encoder')
            print_log(
                'Saving visual_encoder image_processor to'
                f'{visual_encoder_path}', 'current')
            image_processor = BUILDER.build(cfg.image_processor)
            image_processor.save_pretrained(visual_encoder_path,
                                            **save_pretrained_kwargs)
            print_log(f'Saving visual_encoder to {visual_encoder_path}',
                      'current')
            self.visual_encoder.save_pretrained(visual_encoder_path,
                                                **save_pretrained_kwargs)

        # Projector
        projector_path = osp.join(save_dir, 'projector')
        print_log(f'Saving projector to {projector_path}', 'current')
        self.projector.save_pretrained(projector_path,
                                       **save_pretrained_kwargs)

    def to_huggingface_llava(self,
                             cfg,
                             save_dir,
                             fp32=False,
                             save_pretrained_kwargs={}):

        LLM_MAPPING = {
            'model': 'language_model.model',
            'lm_head': 'language_model.lm_head',
        }
        VIT_MAPPING = {
            'vision_model': 'vision_tower.vision_model',
        }
        PROJECTOR_MAPPING = {
            'model.0': 'multi_modal_projector.linear_1',
            'model.2': 'multi_modal_projector.linear_2',
        }

        assert getattr(self.llm, 'hf_quantizer', None) is None, \
            'This conversion format does not support quantized LLM.'

        # get state_dict
        llm = self.llm
        if self.use_llm_lora:
            llm = self.llm.merge_and_unload()
        llm.config.use_cache = True
        if not fp32:
            print_log('Convert LLM to float16', 'current')
            llm.half()

        assert isinstance(llm, LlamaForCausalLM), \
            'This conversion format only supports LlamaForCausalLM.'
        llm_state_dict = llm.state_dict()
        llm_state_dict = convert_state_dict_to_hf(llm_state_dict, LLM_MAPPING)

        need_visual_encoder = (not self.freeze_visual_encoder
                               or self.use_visual_encoder_lora)
        visual_encoder = self.visual_encoder
        if self.use_visual_encoder_lora:
            visual_encoder = self.visual_encoder.merge_and_unload()
        assert isinstance(visual_encoder, CLIPVisionModel),\
            'This conversion format only supports CLIPVisionModel.'
        if need_visual_encoder:
            visual_encoder_state_dict = visual_encoder.state_dict()
            visual_encoder_state_dict = convert_state_dict_to_hf(
                visual_encoder_state_dict, VIT_MAPPING)
        else:
            visual_encoder_state_dict = {}

        projector_state_dict = self.projector.state_dict()
        projector_state_dict = convert_state_dict_to_hf(
            projector_state_dict, PROJECTOR_MAPPING)

        state_dict = {
            **projector_state_dict,
            **llm_state_dict,
            **visual_encoder_state_dict
        }

        # init model
        text_config = llm.config
        vision_config = visual_encoder.config
        config = LlavaConfig(
            text_config=text_config,
            vision_config=vision_config,
            attn_implementation='eager')

        with init_empty_weights():
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    'ignore', message='.*non-meta.*', category=UserWarning)
                model = LlavaForConditionalGeneration(config)
        model.load_state_dict(state_dict, strict=True, assign=True)

        # processor
        cfg.tokenizer.type = LlamaTokenizerFast.from_pretrained
        tokenizer = BUILDER.build(cfg.tokenizer)

        tokenizer.add_tokens(
            AddedToken(DEFAULT_IMAGE_TOKEN, special=True, normalized=False),
            special_tokens=True)
        tokenizer.add_special_tokens({'pad_token': '<pad>'})

        image_processor = BUILDER.build(cfg.image_processor)
        assert isinstance(image_processor, CLIPImageProcessor),\
            'This conversion format only supports CLIPImageProcessor.'

        processor = LlavaProcessor(
            tokenizer=tokenizer, image_processor=image_processor)

        # Pad to 64 for performance reasons
        pad_shape = 64

        pre_expansion_embeddings = \
            model.language_model.model.embed_tokens.weight.data
        mu = torch.mean(pre_expansion_embeddings, dim=0).float()
        n = pre_expansion_embeddings.size()[0]
        sigma = ((pre_expansion_embeddings - mu).T
                 @ (pre_expansion_embeddings - mu)) / n
        dist = torch.distributions.multivariate_normal.MultivariateNormal(
            mu, covariance_matrix=1e-5 * sigma)

        # We add an image token so we need to resize the model
        ori_vocab_size = config.text_config.vocab_size
        tokenizer_vocab_size = tokenizer.encode('<pad>')[-1]
        added_token = tokenizer_vocab_size - ori_vocab_size

        if added_token > 0:
            model.resize_token_embeddings(ori_vocab_size + added_token,
                                          pad_shape)
            model.language_model.model.embed_tokens.weight.data[
                ori_vocab_size:] = torch.stack(
                    tuple(
                        dist.sample()
                        for _ in range(model.language_model.model.embed_tokens.
                                       weight.data[ori_vocab_size:].shape[0])),
                    dim=0,
                )
            model.language_model.lm_head.weight.data[
                ori_vocab_size:] = torch.stack(
                    tuple(dist.sample()
                          for _ in range(model.language_model.lm_head.weight.
                                         data[ori_vocab_size:].shape[0])),
                    dim=0,
                )
        model.config.image_token_index = tokenizer.encode(
            DEFAULT_IMAGE_TOKEN)[-1]
        model.config.pad_token_id = tokenizer.encode('<pad>')[-1]

        # save
        print_log(f'Saving to {save_dir}', 'current')
        model.save_pretrained(save_dir, **save_pretrained_kwargs)
        processor.save_pretrained(save_dir, **save_pretrained_kwargs)

    def to_official_llava(self,
                          cfg,
                          save_dir,
                          fp32=False,
                          save_pretrained_kwargs={}):

        VIT_MAPPING = {
            'vision_model': 'model.vision_tower.vision_tower.vision_model',
        }
        PROJECTOR_MAPPING = {
            'model.0': 'model.mm_projector.0',
            'model.2': 'model.mm_projector.2',
        }

        try:
            from llava.model import LlavaConfig, LlavaLlamaForCausalLM
        except ImportError:
            raise ImportError(
                'Please install llava with '
                '`pip install git+https://github.com/haotian-liu/LLaVA.git '
                '--no-deps`.')

        assert getattr(self.llm, 'hf_quantizer', None) is None, \
            'This conversion format does not support quantized LLM.'

        # get state_dict
        llm = self.llm
        if self.use_llm_lora:
            llm = self.llm.merge_and_unload()
        llm.config.use_cache = True
        if not fp32:
            print_log('Convert LLM to float16', 'current')
            llm.half()

        assert isinstance(llm, LlamaForCausalLM), \
            'This conversion format only supports LlamaForCausalLM.'
        llm_state_dict = llm.state_dict()

        need_visual_encoder = (not self.freeze_visual_encoder
                               or self.use_visual_encoder_lora)
        visual_encoder = self.visual_encoder
        if self.use_visual_encoder_lora:
            visual_encoder = self.visual_encoder.merge_and_unload()
        assert isinstance(visual_encoder, CLIPVisionModel),\
            'This conversion format only supports CLIPVisionModel.'
        if need_visual_encoder:
            visual_encoder_state_dict = visual_encoder.state_dict()
            visual_encoder_state_dict = convert_state_dict_to_hf(
                visual_encoder_state_dict, VIT_MAPPING)
        else:
            visual_encoder_state_dict = {}

        projector_state_dict = self.projector.state_dict()
        projector_state_dict = convert_state_dict_to_hf(
            projector_state_dict, PROJECTOR_MAPPING)

        state_dict = {
            **projector_state_dict,
            **llm_state_dict,
            **visual_encoder_state_dict
        }

        # init model
        tokenizer = BUILDER.build(cfg.tokenizer)
        image_processor = BUILDER.build(cfg.image_processor)
        assert isinstance(image_processor, CLIPImageProcessor),\
            'This conversion format only supports CLIPImageProcessor.'

        llava_config_dict = llm.config.__dict__.copy()
        llava_config_dict.update(
            dict(
                image_aspect_ratio='pad',
                mm_hidden_size=visual_encoder.config.hidden_size,
                mm_projector_type=f'mlp{self.projector_depth}x_gelu',
                mm_use_im_patch_token=False,
                mm_use_im_start_end=False,
                mm_vision_select_feature='patch',
                mm_vision_select_layer=self.visual_select_layer,
                mm_vision_tower=visual_encoder.config.name_or_path,
                unfreeze_mm_vision_tower=need_visual_encoder,
                model_type='llava',
                use_cache=True,
                use_mm_proj=True))

        llava_config = LlavaConfig(**llava_config_dict)

        with init_empty_weights():
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    'ignore', message='.*non-meta.*', category=UserWarning)
                model = LlavaLlamaForCausalLM(llava_config)

        model.load_state_dict(state_dict, strict=True, assign=True)

        # save
        print_log(f'Saving to {save_dir}', 'current')

        model.save_pretrained(save_dir, **save_pretrained_kwargs)
        image_processor.save_pretrained(save_dir, **save_pretrained_kwargs)
        tokenizer.save_pretrained(save_dir, **save_pretrained_kwargs)
    

class AuroraAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        assert (
            self.head_dim * self.num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        size: torch.Tensor = None,  # weight after token merge
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        # get token merge metric
        metric = key_states.view(bsz, self.num_heads, -1, self.head_dim).mean(dim=1)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )

        # apply the causal_attention_mask first
        if causal_attention_mask is not None:
            if causal_attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {causal_attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + causal_attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {causal_attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if size is not None:
            attn_weights = attn_weights + size.log().repeat(self.num_heads, 1, 1)

        attn_weights = F.softmax(attn_weights, dim=-1)

        if output_attentions:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, metric


class AuroraCLIPEncoderLayer(nn.Module):
    def __init__(self, config: CLIPConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = AuroraAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.mlp = CLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        r: int,
        size: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
        output_attentions: bool = False,
    ):
        """
        Args:
            hidden_states (:obj:`torch.FloatTensor`): input to the layer of shape :obj:`(seq_len, batch, embed_dim)`
            attention_mask (:obj:`torch.FloatTensor`): attention mask of size
                :obj:`(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (:obj:`torch.FloatTensor`): mask for attention heads in a given layer of size
                :obj:`(config.encoder_attention_heads,)`.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights, metric = self.self_attn(
            hidden_states=hidden_states,
            size=size,  # token merge weight
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        # apply token merge and update token size as well
        merge, _ = bipartite_soft_matching(metric, r=r, class_token=True, distill_token=False)
        hidden_states, size = merge_wavg(merge, hidden_states, size)

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs, size


class AuroraCLIPEncoder(nn.Module):
    def __init__(self, config: CLIPConfig, r):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([AuroraCLIPEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = True

        # token merge ratio
        self.r = r

    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            causal_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Causal mask for the text model. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        # reset token size for token merge
        size = None

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_states = inputs_embeds 

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs, size = self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    self.r,
                    size,
                    attention_mask,
                    causal_attention_mask,
                    output_attentions,
                )
            else:
                layer_outputs, size = encoder_layer(
                    hidden_states,
                    self.r,
                    size,
                    attention_mask,
                    causal_attention_mask,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)

        # finally we select -2 layer output without cls token

        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions)


class AuroraCLIPVisionTransformer(CLIPVisionTransformer):
    def __init__(self, config: CLIPVisionConfig, r):
        super().__init__(config)
        self.encoder = AuroraCLIPEncoder(config, r)


class AuroraEncoder(CLIPVisionModel):
    def __init__(self, config: CLIPVisionConfig, visual_token_merge_ratio=1):
        super().__init__(config)
        self.vision_model = AuroraCLIPVisionTransformer(config, visual_token_merge_ratio)
        self.visual_token_merge_ratio = visual_token_merge_ratio
        # Initialize weights and apply final processing
        self.post_init()

        # keep the original position embedding
        self.pos_emb = self.vision_model.embeddings.position_embedding.weight

    def reset_tome_r(self, visual_token_merge_ratio):
        self.visual_token_merge_ratio = visual_token_merge_ratio
    
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        self.interpolate_pos_encoding(pixel_values)
        
        # compute and reset token merge number r
        r = int(pixel_values.shape[-1] * pixel_values.shape[-2] / (self.config.patch_size**2) * (1 - self.visual_token_merge_ratio) / self.config.num_hidden_layers)
        self.vision_model.encoder.r = r

        pixel_values = pixel_values.to(self.pos_emb.device)
        return self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )



    # modified from https://github.com/facebookresearch/dino/blob/main/vision_transformer.py#L174
    def interpolate_pos_encoding(self, pixel_values):
        pos_embed = self.pos_emb
        device = pos_embed.device

        patch_size = self.vision_model.config.patch_size
        w0 = pixel_values.shape[-2] // patch_size
        h0 = pixel_values.shape[-1] // patch_size
        npatch = w0 * h0
        N = pos_embed.shape[0] - 1

        if npatch == N and w0 == h0:
            # no need to interpolate
            # reset the position embedding to the original position embedding
            self.vision_model.embeddings.position_embedding.weight = nn.Parameter(pos_embed).to(device)
            self.vision_model.embeddings.position_ids = torch.arange(pos_embed.shape[0]).to(device)
            return

        class_pos_embed = pos_embed[0]
        patch_pos_embed = pos_embed[1:]
        dim = class_pos_embed.shape[-1]

        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode="bicubic",
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        # print(f'Interpolated position encoding to match input size: {patch_pos_embed.shape[-2], patch_pos_embed.shape[-1]}')

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(-1, dim)

        interpolated_pos_embed = torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=0)

        self.vision_model.embeddings.position_embedding.weight = nn.Parameter(interpolated_pos_embed).to(device)
        self.vision_model.embeddings.position_ids = (
            torch.arange(interpolated_pos_embed.shape[0]).reshape(1, -1).to(device)
        )

        return


class AuroraSIGLIPEncoderLayer(nn.Module):
    def __init__(self, config: SiglipConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = AuroraAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        r: int,
        size: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
        output_attentions: bool = False,
    ):
        """
        Args:
            hidden_states (:obj:`torch.FloatTensor`): input to the layer of shape :obj:`(seq_len, batch, embed_dim)`
            attention_mask (:obj:`torch.FloatTensor`): attention mask of size
                :obj:`(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (:obj:`torch.FloatTensor`): mask for attention heads in a given layer of size
                :obj:`(config.encoder_attention_heads,)`.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights, metric = self.self_attn(
            hidden_states=hidden_states,
            size=size,  # token merge weight
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        # apply token merge and update token size as well
        merge, _ = bipartite_soft_matching(metric, r=r, class_token=True, distill_token=False)
        hidden_states, size = merge_wavg(merge, hidden_states, size)

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs, size


class AuroraSIGLIPEncoder(nn.Module):
    def __init__(self, config: SiglipConfig, r):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([AuroraSIGLIPEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = True

        # token merge ratio
        self.r = r


    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            causal_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Causal mask for the text model. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        # reset token size for token merge
        size = None

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        hidden_states = inputs_embeds

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs, size = self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    self.r,
                    size,
                    attention_mask,
                    causal_attention_mask,
                    output_attentions,
                )
            else:
                layer_outputs, size = encoder_layer(
                    hidden_states,
                    self.r,
                    size,
                    attention_mask,
                    causal_attention_mask,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)

        # finally we select -2 layer output without cls token

        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions)

class AuroraSIGLIPVisionTransformer(SiglipVisionTransformer):
    def __init__(self, config: SiglipVisionConfig, r):
        super().__init__(config)
        self.encoder = AuroraSIGLIPEncoder(config, r)

class AuroraSigEncoder(SiglipVisionModel):
    def __init__(self, config: SiglipVisionConfig, visual_token_merge_ratio=1):
        super().__init__(config)
        self.vision_model = AuroraSIGLIPVisionTransformer(config, visual_token_merge_ratio)
        self.visual_token_merge_ratio = visual_token_merge_ratio
        # Initialize weights and apply final processing
        self.post_init()

        # keep the original position embedding
        self.pos_emb = self.vision_model.embeddings.position_embedding.weight

    def reset_tome_r(self, visual_token_merge_ratio):
        self.visual_token_merge_ratio = visual_token_merge_ratio
    
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        self.interpolate_pos_encoding(pixel_values)
        
        # compute and reset token merge number r
        r = int(pixel_values.shape[-1] * pixel_values.shape[-2] / (self.config.patch_size**2) * (1 - self.visual_token_merge_ratio) / self.config.num_hidden_layers)
        self.vision_model.encoder.r = r

        pixel_values = pixel_values.to(self.vision_model.embeddings.position_embedding.weight.dtype).to(self.vision_model.embeddings.position_embedding.weight.dtype)
        return self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    # modified from https://github.com/facebookresearch/dino/blob/main/vision_transformer.py#L174
    def interpolate_pos_encoding(self, pixel_values):
        pos_embed = self.pos_emb
        device = pos_embed.device

        patch_size = self.vision_model.config.patch_size
        w0 = pixel_values.shape[-2] // patch_size
        h0 = pixel_values.shape[-1] // patch_size
        npatch = w0 * h0
        N = pos_embed.shape[0] 

        if npatch == N and w0 == h0:
            # no need to interpolate
            # reset the position embedding to the original position embedding
            self.vision_model.embeddings.position_embedding.weight = nn.Parameter(pos_embed).to(device)
            self.vision_model.embeddings.position_ids = torch.arange(pos_embed.shape[0]).to(device)
            return

        class_pos_embed = pos_embed[0]
        patch_pos_embed = pos_embed[1:]
        dim = class_pos_embed.shape[-1]

        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode="bicubic",
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        # print(f'Interpolated position encoding to match input size: {patch_pos_embed.shape[-2], patch_pos_embed.shape[-1]}')

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(-1, dim)

        interpolated_pos_embed = torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=0)

        self.vision_model.embeddings.position_embedding.weight = nn.Parameter(interpolated_pos_embed).to(device)
        self.vision_model.embeddings.position_ids = (
            torch.arange(interpolated_pos_embed.shape[0]).reshape(1, -1).to(device)
        )

        return


