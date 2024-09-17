from transformers import LlavaForConditionalGeneration as HF_LlavaForConditionalGeneration
from transformers.models.llava.modeling_llava import LlavaCausalLMOutputWithPast
import torch
from typing import List, Optional, Tuple, Union
from torch import nn
import math
import torch.distributed as dist
from xtuner._lite.parallel import (LengthGroupedSampler, ParallelSampler,
                                   get_dp_mesh, get_dp_world_size,
                                   get_sp_group, get_sp_mesh,
                                   get_sp_world_size,
                                   reduce_sequence_parallel_loss,
                                   setup_parallel, get_sp_group, split_for_sequence_parallel)

from mmengine import MessageHub


class _GatherFromSeqParallelRegion(torch.autograd.Function):
    """Gather the input from model parallel region and concatinate."""

    @staticmethod
    def forward(ctx, input_, dim):  # gather
        ctx.dim = dim
        sp_group = get_sp_group()
        sp_world_size = get_sp_world_size()
        if sp_world_size == 1:
            return input_

        tensor_list = [torch.empty_like(input_) for _ in range(sp_world_size)]
        torch.distributed.all_gather(tensor_list, input_, group=sp_group)

        # Note: torch.cat already creates a contiguous tensor.
        output = torch.cat(tensor_list, dim=dim).contiguous()
        return output

    @staticmethod
    def backward(ctx, grad_output):  # split
        sp_group = get_sp_group()
        sp_rank = dist.get_rank(sp_group)
        sp_world_size = get_sp_world_size()
        if sp_world_size == 1:
            return grad_output

        # Split along last dimension.
        last_dim_size = grad_output.size()[ctx.dim] // sp_world_size
        tensor_list = torch.split(grad_output, last_dim_size, dim=ctx.dim)
        output = tensor_list[sp_rank].contiguous()
        return output, None


class LlavaForConditionalGeneration(HF_LlavaForConditionalGeneration):
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            pixel_values: torch.FloatTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            vision_feature_layer: Optional[int] = None,
            vision_feature_select_strategy: Optional[str] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, LlavaCausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )

        sp_size = get_sp_world_size()

        if inputs_embeds is None:
            # 1. Extra the input embeddings
            sp_group = get_sp_group()
            sp_rank = dist.get_rank(sp_group)
            if sp_size > 1:
                orig_len_input_ids = input_ids.shape[1]
                assert input_ids.shape[0] == 1, 'batch size must be 1 for sequence parallel'
                # input_ids 均匀切分
                if orig_len_input_ids % sp_size != 0:  # 确保能均匀切
                    max_inputs_len = math.ceil(orig_len_input_ids / sp_size) * sp_size
                    _temp = torch.full((1, max_inputs_len - orig_len_input_ids), self.config.pad_token_id,
                                       dtype=input_ids.dtype,
                                       device=input_ids.device)
                    input_ids = torch.cat([input_ids, _temp], dim=-1)
                input_ids = torch.split(input_ids, input_ids.shape[1] // sp_size, dim=-1)
                input_ids = input_ids[sp_rank].contiguous()
            inputs_embeds = self.get_input_embeddings()(input_ids)
            if sp_size > 1:
                # 重新合并
                inputs_embeds = _GatherFromSeqParallelRegion.apply(inputs_embeds, 1)
                # tensor_list = [torch.empty_like(inputs_embeds) for _ in range(sp_size)]
                # dist.all_gather(tensor_list, inputs_embeds, group=sp_group)
                # inputs_embeds = torch.cat(tensor_list, dim=1).contiguous()
                # 移除原始的pad
                inputs_embeds = inputs_embeds[:, :orig_len_input_ids]

                input_ids = _GatherFromSeqParallelRegion.apply(input_ids, 1)
                # tensor_list = [torch.empty_like(input_ids) for _ in range(sp_size)]
                # dist.all_gather(tensor_list, input_ids, group=sp_group)
                # input_ids = torch.cat(tensor_list, dim=1).contiguous()
                # 移除原始的pad
                input_ids = input_ids[:, :orig_len_input_ids].contiguous()
            # ------------- start add this ----------------
            if pixel_values is None:
                # all of the input is text
                # If not handled properly, deadlock can occur.
                # print('===================all of the input is text==============')
                image_size = self.config.vision_config.image_size
                pixel_values = torch.zeros(input_ids.shape[0], 3, image_size, image_size,
                                           dtype=torch.float32,
                                           device=input_ids.device)
                image_outputs = self.vision_tower(pixel_values, output_hidden_states=True)
                # this is not memory efficient at all (output_hidden_states=True) will save all the hidden stated.
                selected_image_feature = image_outputs.hidden_states[vision_feature_layer]
                if vision_feature_select_strategy == "default":
                    selected_image_feature = selected_image_feature[:, 1:]
                elif vision_feature_select_strategy == "full":
                    selected_image_feature = selected_image_feature
                else:
                    raise ValueError(
                        f"Unexpected select feature strategy: {self.config.vision_feature_select_strategy}"
                    )
                image_features = self.multi_modal_projector(selected_image_feature)
                inputs_embeds = inputs_embeds.to(image_features.dtype)
                inputs_embeds, attention_mask, labels, position_ids = self._merge_input_ids_with_image_features(
                    image_features[0:0], inputs_embeds, input_ids, attention_mask, labels
                )
            # ------------- end add this ----------------
            # 2. Merge text and images
            elif pixel_values is not None and input_ids.shape[1] != 1:
                # 图片均匀切分
                if sp_size > 1:
                    # pixel_values 均匀切分
                    orig_img_batch = pixel_values.shape[0]
                    if orig_img_batch % sp_size != 0:  # 确保能均匀切
                        max_inputs_len = math.ceil(orig_img_batch / sp_size) * sp_size
                        pad_img_batch = max_inputs_len - orig_img_batch
                        pad_pixel_values_ = torch.zeros(pad_img_batch, 3,
                                                        pixel_values.shape[2],
                                                        pixel_values.shape[3],
                                                        dtype=pixel_values.dtype,
                                                        device=pixel_values.device)

                        pixel_values = torch.cat([pixel_values, pad_pixel_values_], dim=0)
                    pixel_values = torch.split(pixel_values, len(pixel_values) // sp_size, dim=0)
                    pixel_values = pixel_values[sp_rank].contiguous()
                image_outputs = self.vision_tower(pixel_values, output_hidden_states=True)
                # this is not memory efficient at all (output_hidden_states=True) will save all the hidden stated.
                selected_image_feature = image_outputs.hidden_states[vision_feature_layer]

                if vision_feature_select_strategy == "default":
                    selected_image_feature = selected_image_feature[:, 1:]
                elif vision_feature_select_strategy == "full":
                    selected_image_feature = selected_image_feature
                else:
                    raise ValueError(
                        f"Unexpected select feature strategy: {self.config.vision_feature_select_strategy}"
                    )

                image_features = self.multi_modal_projector(selected_image_feature)
                inputs_embeds = inputs_embeds.to(image_features.dtype)

                # 切分后合并
                if sp_size > 1:
                    # 重新合并
                    image_features = _GatherFromSeqParallelRegion.apply(image_features, 0)
                    # tensor_list = [torch.empty_like(image_features) for _ in range(sp_size)]
                    # dist.all_gather(tensor_list, image_features, group=sp_group)
                    # image_features = torch.cat(tensor_list, dim=0).contiguous()
                    # 移除多余的pad
                    image_features = image_features[:orig_img_batch]
                inputs_embeds, attention_mask, labels, position_ids = self._merge_input_ids_with_image_features(
                    image_features, inputs_embeds, input_ids, attention_mask, labels
                )

            # In case input_ids.shape[1] == 1 & pixel_values==None & past_key_values != None, we are in the case of
            # generation with cache
            elif past_key_values is not None and pixel_values is not None and input_ids.shape[1] == 1:
                # Retrieve the first layer to inspect the logits and mask out the hidden states
                # that are set to 0
                first_layer_past_key_value = past_key_values[0][0][:, :, :, 0]

                # Sum all dimensions of head_dim (-2) to avoid random errors such as: https://github.com/huggingface/transformers/pull/28032#issuecomment-1863691941
                batch_index, non_attended_tokens = torch.where(first_layer_past_key_value.float().sum(-2) == 0)

                # Get the target length
                target_length = input_ids.shape[1]
                past_length = first_layer_past_key_value.shape[-1]

                extended_attention_mask = torch.ones(
                    (attention_mask.shape[0], past_length),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )

                # Filter out only the tokens that can be un-attended, this can happen
                # if one uses Llava + Fused modules where the cache on the
                # first iteration is already big enough, or if one passes custom cache
                valid_indices = non_attended_tokens < extended_attention_mask.size(-1)
                new_batch_index = batch_index[valid_indices]
                new_non_attended_tokens = non_attended_tokens[valid_indices]

                # Zero-out the places where we don't need to attend
                extended_attention_mask[new_batch_index, new_non_attended_tokens] = 0

                attention_mask = torch.cat((extended_attention_mask, attention_mask[:, -target_length:]), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1

        # 此处开始进行切分处理
        # 只需要处理 inputs_embeds 和 position_ids，其余用不到
        attn_context = MessageHub.get_instance('packed_sequence')
        position_ids = attn_context.get_info('position_ids')
        assert position_ids.size(1) == inputs_embeds.shape[1] == labels.shape[1], \
            f'{position_ids.size(1)} {inputs_embeds.shape[1]} {labels.shape[1]}'

        if sp_size > 1:
            sp_group = get_sp_group()
            # `dim` is 1 as the shape of tensor is (bs, seq_len)
            position_ids = split_for_sequence_parallel(
                position_ids, dim=1, sp_group=sp_group)
            inputs_embeds = split_for_sequence_parallel(
                inputs_embeds, dim=1, sp_group=sp_group)
            labels = split_for_sequence_parallel(
                labels, dim=1, sp_group=sp_group)
            attention_mask = None  # 不需要
            attn_context.update_info('position_ids', position_ids)
        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = outputs[0]

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                shift_attention_mask = attention_mask[..., 1:]
                shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return LlavaCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
