from modeling_mixformer_sequential import MixFormerSequentialForCausalLM, InferenceParams
from configuration_mixformer_sequential import MixFormerSequentialConfig

from transformers.modeling_outputs import CausalLMOutputWithPast
import torch
from typing import Any, Dict, Optional, Tuple, Union

class ClientSideMixFormerSequentialForCausalLM(MixFormerSequentialForCausalLM):
    
    def __init__(self, config):
        super().__init__(config)
        self.split_layer=2
        
    def forward(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Union[torch.FloatTensor, InferenceParams]] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        if attention_mask is not None and self.training:
            print("`attention_mask` is not supported during training. Using it might lead to unexpected results.")

        if past_key_values is None and attention_mask is None:
            print("past_key_values & attention_mask is None!")
            lm_logits = self.layers(input_ids)
            return lm_logits
        else:
            print("forward with past_key_values or attention_mask!")
            hidden_layer = self.layers[0](input_ids)
            for module in self.layers[1: self.split_layer]:  # return intermediate tensor
                hidden_layer = module(hidden_layer, past_key_values=past_key_values, attention_mask=attention_mask)
            return input_ids, past_key_values, attention_mask, labels, hidden_layer



class ServerSideMixFormerSequentialForCausalLM(MixFormerSequentialForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.split_layer=2
        
    def forward(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Union[torch.FloatTensor, InferenceParams]] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        hidden_layer_input: torch.Tensor = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        if attention_mask is not None and self.training:
            print("`attention_mask` is not supported during training. Using it might lead to unexpected results.")

        if past_key_values is None and attention_mask is None:
            lm_logits = self.layers(input_ids)
        else:
            hidden_layer = hidden_layer_input
            for module in self.layers[self.split_layer:-1]:  # Compute the remaining block 
                hidden_layer = module(hidden_layer, past_key_values=past_key_values, attention_mask=attention_mask)
            lm_logits = self.layers[-1](hidden_layer)
            
        loss = None
        if labels is not None:
            loss = self.loss(lm_logits, labels)

        return CausalLMOutputWithPast(loss=loss, logits=lm_logits, past_key_values=past_key_values)

