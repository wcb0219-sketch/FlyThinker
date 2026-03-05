# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
import torch.nn as nn
from collections import namedtuple
from FlyThinker.Train.utils.modeling_qwen2 import Qwen2ForCausalLM
from transformers import AutoConfig

Outputs = namedtuple("Outputs", ["loss","loss_orig","loss_diff","kl_loss"])
MAX_N_LATENT = 8

class BaseModelOutputWithPast:
    def __init__(self,output):
        self.logits = output['logits']
        self.past_key_values = output['past_key_values']
        self.hidden_states = output['hidden_states']
                
class FlyThinker(nn.Module):

    def __init__(
        self,
        base_causallm,
        reasoner_path = None,
    ):

        super(FlyThinker, self).__init__()
        self.gen_forward_cnt = 0
        self.Generator = base_causallm
        self.config = self.Generator.config
        self.reasoner_path = reasoner_path
        
        self.loss_fct = nn.CrossEntropyLoss(reduction='none')

        print(f'reasoner_path:{reasoner_path}')
        config = AutoConfig.from_pretrained(reasoner_path, trust_remote_code=True)
        self.Reasoner = Qwen2ForCausalLM.from_pretrained(reasoner_path,
                                                            config = config,
                                                            dtype=torch.bfloat16,
                                                            # attn_implementation='flash_attention_2',
                                                            trust_remote_code=True)
        
        embedding = self.Reasoner.get_input_embeddings()
        
        # for name, param in self.Reasoner.named_parameters():
        #     param.requires_grad = False
        # self.Reasoner.load_state_dict(torch.load('/home/hexngroup/wangcbing/verl/lamp_ckpt/latent_none_2e-5_bz64_new/global_step_1000_coconut_True_vae_None/model.pth'))
            
        
    def fixed_cross_entropy(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        num_items_in_batch = None,
        ignore_index: int = -100,
        **kwargs,
    ) -> torch.Tensor:
        reduction = "sum" if num_items_in_batch is not None else "mean"
        loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
        if reduction == "sum":
            if torch.is_tensor(num_items_in_batch):
                num_items_in_batch = num_items_in_batch.to(loss.device)
            loss = loss / num_items_in_batch
        return loss

    def ForCausalLMLoss(
        self,
        logits,
        labels,
        vocab_size: int,
        num_items_in_batch = None,
        ignore_index = -100,
        shift_labels = None,
        **kwargs,
    ) -> torch.Tensor:
        logits = logits.float()

        if shift_labels is None:
            labels = nn.functional.pad(labels, (0, 1), value=ignore_index)
            shift_labels = labels[..., 1:].contiguous()

        logits = logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1)
        shift_labels = shift_labels.to(logits.device)
        loss = self.fixed_cross_entropy(logits, shift_labels, num_items_in_batch, ignore_index, **kwargs)
        return loss
    

    def forward(self, input_ids, attention_mask, labels, position_ids, diff_lambda = 0.1, **kwargs):
        

        input_emb = self.Generator.get_input_embeddings()(input_ids)
        input_emb_norms = torch.norm(self.Generator.get_input_embeddings().weight, p=2, dim=-1).mean().detach()

        output_r1 = self.Reasoner(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    labels=None,
                                    position_ids=position_ids,
                                    use_cache=True,
                                    output_hidden_states = False,
                                    )
       
        hidden_states = output_r1.hidden_states[:,:-1,:]

        hidden_norms = torch.norm(hidden_states, p=2, dim=-1).detach()
        hidden_states = hidden_states * input_emb_norms / hidden_norms[:,:,None]
        
        if input_emb.shape[-1] == hidden_states.shape[-1]:
            insert_emb = input_emb[:,1:,:] + diff_lambda * hidden_states
        else:
            input_emb[:,1:,:hidden_states.shape[-1]] = input_emb[:,1:,:hidden_states.shape[-1]] + diff_lambda * hidden_states
            insert_emb = input_emb[:,1:,:]
        
        draft_emb = torch.cat((input_emb[:,:1,:],insert_emb), dim = 1)

        output_r2 = self.Generator(inputs_embeds=draft_emb,
                                        attention_mask=attention_mask,
                                        labels=None,
                                        position_ids=position_ids,
                                        use_cache=True,
                                        )
        
        loss = self.ForCausalLMLoss(logits=output_r2.logits, labels=labels, vocab_size=self.Generator.config.vocab_size, **kwargs)
        
        return loss

    def train(self, mode: bool = True):
        super().train(mode)                
        self.Generator.train(mode)
        self.Reasoner.train(mode)
    
    def eval(self, mode: bool = True):
        super().eval(mode)                
        self.Generator.eval(mode)     
        self.Reasoner.eval(mode)      

