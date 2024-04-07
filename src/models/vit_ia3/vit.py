#!/usr/bin/env python3
"""
vit with ia3
"""
import copy
import numpy as np
import torch
import torch.nn as nn

from torch.nn import Parameter, Linear, LayerNorm
from ..vit_backbones.vit import *
from ...utils import logging
logger = logging.get_logger("PEFT")


class IA3Attention(Attention):
    def __init__(self, config, ia3_config, vis):
        super(IA3Attention, self).__init__(config, vis)

        self.ia3_config = ia3_config

        self.key_ia3 = Parameter(torch.ones(self.all_head_size))
        self.value_ia3 = Parameter(torch.ones(self.all_head_size))

        nn.init.normal_(self.key_ia3, mean=1.0, std=0.02)
        nn.init.normal_(self.value_ia3, mean=1.0, std=0.02)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states) # B, num_patches, head_size*num_head
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        mixed_key_layer = mixed_key_layer * self.key_ia3
        mixed_value_layer = mixed_value_layer * self.value_ia3

        query_layer = self.transpose_for_scores(mixed_query_layer) # B, num_head, num_patches, head_size
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) # B, num_head, num_patches, num_patches
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores) # B, num_head, num_patches(query), num_patches(key)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer) # B, num_head, num_patches, head_size
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class IA3Mlp(Mlp):
    def __init__(self, config, ia3_config):
        super(IA3Mlp, self).__init__(config)

        self.ia3_config = ia3_config

        self.ffn_ia3 = Parameter(torch.ones(config.transformer["mlp_dim"]))
        nn.init.normal_(self.ffn_ia3, mean=1.0, std=0.02)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = x * self.ffn_ia3
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class IA3Block(Block):
    def __init__(self, config, ia3_config, vis):
        super(IA3Block, self).__init__(config, vis)

        self.ia3_config = ia3_config

        self.attn = IA3Attention(config, ia3_config, vis)
        self.ffn = IA3Mlp(config, ia3_config)


class IA3Encoder(Encoder):
    def __init__(self, config, ia3_cfg, vis):
        super(IA3Encoder, self).__init__(config, vis)

        self.layer = nn.ModuleList()
        self.num_layers = config.transformer["num_layers"]
        for _ in range(self.num_layers):
            layer = IA3Block(config, ia3_cfg, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class IA3Transformer(Transformer):
    def __init__(self, config, ia3_cfg, img_size, vis):
        super(IA3Transformer, self).__init__(config, img_size, vis)

        self.vit_config = config
        self.ia3_cfg = ia3_cfg
        self.encoder = IA3Encoder(config, ia3_cfg, vis)

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)

        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights


class IA3VisionTransformer(VisionTransformer):
    def __init__(
        self, model_type, img_size=224, num_classes=21843, ia3_cfg=None, vis=False
    ):
        super(IA3VisionTransformer, self).__init__(model_type, img_size, num_classes, vis)

        vit_cfg = CONFIGS[model_type]
        if ia3_cfg is None:
            raise ValueError("ia3_cfg cannot be None if using IA3VisionTransformer")

        self.ia3_cfg = ia3_cfg
        self.transformer = IA3Transformer(vit_cfg, ia3_cfg, img_size, vis)

    def forward(self, x, vis=False):
        x, attn_weights = self.transformer(x)
        logits = self.head(x[:, 0])

        if not vis:
            return logits
        return logits, attn_weights
