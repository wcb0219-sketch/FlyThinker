# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig

class FlyThinker(nn.Module):

    def __init__(
        self,
        base_causallm,
        reasoner_path = None,
    ):

        super(FlyThinker, self).__init__()
        self.gen_forward_cnt = 0
        self.Generator = base_causallm
        config = AutoConfig.from_pretrained(reasoner_path, trust_remote_code=True)

        self.Reasoner = AutoModelForCausalLM.from_pretrained(reasoner_path,
                                                            config = config,
                                                            dtype=torch.bfloat16,
                                                            # attn_implementation='flash_attention_2',
                                                            trust_remote_code=True)#.cuda()

    
    def train(self):
        self.Generator.train()
        self.Reasoner.train()

    def eval(self):
        self.Generator.eval()
        self.Reasoner.eval()