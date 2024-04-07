#!/usr/bin/env python3
"""
vit with adapter
"""
import copy
import numpy as np
import torch
import torch.nn as nn

from torch.nn import Linear, LayerNorm
from ..vit_backbones.vit import *
from ...utils import logging
logger = logging.get_logger("PEFT")


class Adapter(nn.Module):
    def __init__(self, dim, adapter_config):
        super(Adapter, self).__init__()

        self.adapter_config = adapter_config
        bottleneck_dim = adapter_config.BOTTLENECK

        self.adapter_down = nn.Linear(dim, bottleneck_dim)
        self.adapter_up = nn.Linear(bottleneck_dim, dim)
        self.adapter_act = ACT2FN[adapter_config.ACTIVATION]
        if adapter_config.LN_BEFORE:
            self.adapter_ln_before = nn.LayerNorm(dim)
        if adapter_config.LN_AFTER:
            self.adapter_ln_after = nn.LayerNorm(dim)

        nn.init.zeros_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)
    
    def forward(self, x):
        if self.adapter_config.LN_BEFORE:
            x = self.adapter_ln_before(x)
        x = self.adapter_down(x)
        x = self.adapter_act(x)
        x = nn.functional.dropout(x, p=self.adapter_config.DROPOUT, training=self.training)
        x = self.adapter_up(x)
        if self.adapter_config.LN_AFTER:
            x = self.adapter_ln_after(x)
        return x


class AdaptedBlock(Block):
    def __init__(self, config, adapter_config, vis):
        super(AdaptedBlock, self).__init__(config, vis)

        self.adapter_config = adapter_config

        if adapter_config.TYPE == "Houlsby":
            self.attn_adapter = Adapter(config.hidden_size, adapter_config)
            self.ffn_adapter = Adapter(config.hidden_size, adapter_config)
        
        elif adapter_config.TYPE == "Pfeiffer" or adapter_config.TYPE == "Chen":
            self.adapter = Adapter(config.hidden_size, adapter_config)
        
        else:
            raise ValueError("Other adapter styles are not supported.")

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)

        if self.adapter_config.TYPE == "Houlsby":
            adpt = self.attn_adapter(x)
            x = x + adpt
        
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)

        if self.adapter_config.TYPE == "Houlsby":
            adpt = self.ffn_adapter(x)
            x = x + adpt
        elif self.adapter_config.TYPE == "Pfeiffer":
            adpt = self.adapter(x)
            x = x + adpt
        elif self.adapter_config.TYPE == "Chen":
            adpt = self.adapter(h)
            x = x + adpt * self.adapter_config.SCALE

        x = x + h 

        return x, weights


class AdaptedEncoder(Encoder):
    def __init__(self, config, adapter_cfg, vis):
        super(AdaptedEncoder, self).__init__(config, vis)

        self.layer = nn.ModuleList()
        self.num_layers = config.transformer["num_layers"]
        for _ in range(self.num_layers):
            layer = AdaptedBlock(config, adapter_cfg, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class AdaptedTransformer(Transformer):
    def __init__(self, config, adapter_cfg, img_size, vis):
        super(AdaptedTransformer, self).__init__(config, img_size, vis)

        self.vit_config = config
        self.adapter_cfg = adapter_cfg
        self.encoder = AdaptedEncoder(config, adapter_cfg, vis)

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)

        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights


class AdaptedVisionTransformer(VisionTransformer):
    def __init__(
        self, model_type, img_size=224, num_classes=21843, adapter_cfg=None, vis=False
    ):
        super(AdaptedVisionTransformer, self).__init__(model_type, img_size, num_classes, vis)

        vit_cfg = CONFIGS[model_type]
        if adapter_cfg is None:
            raise ValueError("adapter_cfg cannot be None if using AdaptedVisionTransformer")

        self.adapter_cfg = adapter_cfg
        self.transformer = AdaptedTransformer(vit_cfg, adapter_cfg, img_size, vis)

    def forward(self, x, vis=False):
        x, attn_weights = self.transformer(x)
        logits = self.head(x[:, 0])

        if not vis:
            return logits
        return logits, attn_weights
