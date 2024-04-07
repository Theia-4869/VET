#!/usr/bin/env python3
"""
vit with prompt
"""
import math
import torch
import torch.nn as nn

from functools import reduce
from operator import mul
from torch.nn import Dropout
from torch.nn.modules.utils import _pair

from ..vit_backbones.vit import CONFIGS, Transformer, VisionTransformer
from ...utils import logging
logger = logging.get_logger("PEFT")


class PromptedTransformer(Transformer):
    def __init__(self, config, prompt_config, img_size, vis):
        assert prompt_config.LOCATION == "prepend"
        assert prompt_config.INITIATION == "random"
        assert prompt_config.NUM_DEEP_LAYERS is None
        assert not prompt_config.DEEP_SHARED
        super(PromptedTransformer, self).__init__(config, img_size, vis)
        
        self.vit_config = config
        self.prompt_config = prompt_config
        
        img_size = _pair(img_size)
        patch_size = _pair(config.patches["size"])

        num_tokens = self.prompt_config.NUM_TOKENS
        self.num_tokens = num_tokens  # number of prompted tokens

        self.prompt_dropout = Dropout(self.prompt_config.DROPOUT)

        # if project the prompt embeddings
        if self.prompt_config.PROJECT > 0:
            # only for prepend / add
            prompt_dim = self.prompt_config.PROJECT
            self.prompt_proj = nn.Linear(prompt_dim, config.hidden_size)
            nn.init.kaiming_normal_(self.prompt_proj.weight, a=0, mode='fan_out')
        else:
            prompt_dim = config.hidden_size
            self.prompt_proj = nn.Identity()

        # initiate prompt:
        if self.prompt_config.INITIATION == "random":
            val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))

            if self.prompt_config.DEEP:
                num_layers = config.transformer["num_layers"]
            else:
                num_layers = 1

            self.prompt_embeddings = nn.Parameter(torch.zeros(num_layers, num_tokens, prompt_dim))
            # xavier_uniform initialization
            nn.init.uniform_(self.prompt_embeddings.data, -val, val)

        else:
            raise ValueError("Other initiation scheme is not supported")

    def forward(self, x):
        hidden_states = self.embeddings(x)

        attn_weights = []
        B = x.shape[0]
        num_layers = self.vit_config.transformer["num_layers"]

        for i in range(num_layers):
            if i == 0:
                if self.prompt_config.DEEP:
                    prompt_emb = self.prompt_dropout(self.prompt_proj(
                        self.prompt_embeddings[i]).expand(B, -1, -1))
                else:
                    prompt_emb = self.prompt_dropout(self.prompt_proj(
                        self.prompt_embeddings).expand(B, -1, -1))
                
                hidden_states = torch.cat((
                    hidden_states[:, :1, :],
                    prompt_emb,
                    hidden_states[:, 1:, :]
                ), dim=1)

            else:
                if self.prompt_config.DEEP:
                    prompt_emb = self.prompt_dropout(self.prompt_proj(
                        self.prompt_embeddings[i-1]).expand(B, -1, -1))

                    hidden_states = torch.cat((
                        hidden_states[:, :1, :],
                        prompt_emb,
                        hidden_states[:, (1+self.num_tokens):, :]
                    ), dim=1)

            hidden_states, weights = self.encoder.layer[i](hidden_states)

            if self.encoder.vis:
                attn_weights.append(weights)

        encoded = self.encoder.encoder_norm(hidden_states)
        return encoded, attn_weights


class PromptedVisionTransformer(VisionTransformer):
    def __init__(
        self, model_type, img_size=224, num_classes=21843, prompt_cfg=None, vis=False
    ):
        assert prompt_cfg.VIT_POOL_TYPE == "original"
        super(PromptedVisionTransformer, self).__init__(model_type, img_size, num_classes, vis)

        vit_cfg = CONFIGS[model_type]
        if prompt_cfg is None:
            raise ValueError("prompt_cfg cannot be None if using PromptedVisionTransformer")
        
        self.prompt_cfg = prompt_cfg
        self.transformer = PromptedTransformer(vit_cfg, prompt_cfg, img_size, vis)

    def forward(self, x, vis=False):
        x, attn_weights = self.transformer(x)
        logits = self.head(x[:, 0])

        if not vis:
            return logits
        return logits, attn_weights
