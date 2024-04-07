#!/usr/bin/env python3
"""
vit with lora
"""
import copy
import numpy as np
import torch
import torch.nn as nn

from torch.nn import Linear, LayerNorm
from ..vit_backbones.vit import *
from ...utils import logging
logger = logging.get_logger("PEFT")


class LoRAAttention(Attention):
    def __init__(self, config, lora_config, vis):
        super(LoRAAttention, self).__init__(config, vis)

        self.lora_config = lora_config
        lora_rank = lora_config.RANK

        if "q" in lora_config.MODE:
            self.query_lora_a = Linear(config.hidden_size, lora_rank, bias=False)
            self.query_lora_b = Linear(lora_rank, self.all_head_size, bias=False)
        if "k" in lora_config.MODE:
            self.key_lora_a = Linear(config.hidden_size, lora_rank, bias=False)
            self.key_lora_b = Linear(lora_rank, self.all_head_size, bias=False)
        if "v" in lora_config.MODE:
            self.value_lora_a = Linear(config.hidden_size, lora_rank, bias=False)
            self.value_lora_b = Linear(lora_rank, self.all_head_size, bias=False)
        if "o" in lora_config.MODE:
            self.output_lora_a = Linear(config.hidden_size, lora_rank, bias=False)
            self.output_lora_b = Linear(lora_rank, config.hidden_size, bias=False)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states) # B, num_patches, head_size*num_head
        if "q" in self.lora_config.MODE:
            mixed_query_layer += self.query_lora_b(self.query_lora_a(hidden_states))
        mixed_key_layer = self.key(hidden_states)
        if "k" in self.lora_config.MODE:
            mixed_key_layer += self.key_lora_b(self.key_lora_a(hidden_states))
        mixed_value_layer = self.value(hidden_states)
        if "v" in self.lora_config.MODE:
            mixed_value_layer += self.value_lora_b(self.value_lora_a(hidden_states))

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
        if "o" in self.lora_config.MODE:
            attention_output += self.output_lora_b(self.output_lora_a(hidden_states))
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class LoRABlock(Block):
    def __init__(self, config, lora_config, vis):
        super(LoRABlock, self).__init__(config, vis)

        self.lora_config = lora_config
        self.attn = LoRAAttention(config, lora_config, vis)


class LoRAEncoder(Encoder):
    def __init__(self, config, lora_cfg, vis):
        super(LoRAEncoder, self).__init__(config, vis)

        self.layer = nn.ModuleList()
        self.num_layers = config.transformer["num_layers"]
        for _ in range(self.num_layers):
            layer = LoRABlock(config, lora_cfg, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class LoRATransformer(Transformer):
    def __init__(self, config, lora_cfg, img_size, vis):
        super(LoRATransformer, self).__init__(config, img_size, vis)

        self.vit_config = config
        self.lora_cfg = lora_cfg
        self.encoder = LoRAEncoder(config, lora_cfg, vis)

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)

        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights


class LoRAVisionTransformer(VisionTransformer):
    def __init__(
        self, model_type, img_size=224, num_classes=21843, lora_cfg=None, vis=False
    ):
        super(LoRAVisionTransformer, self).__init__(model_type, img_size, num_classes, vis)

        vit_cfg = CONFIGS[model_type]
        if lora_cfg is None:
            raise ValueError("lora_cfg cannot be None if using LoRAVisionTransformer")

        self.lora_cfg = lora_cfg
        self.transformer = LoRATransformer(vit_cfg, lora_cfg, img_size, vis)

    def forward(self, x, vis=False):
        x, attn_weights = self.transformer(x)
        logits = self.head(x[:, 0])

        if not vis:
            return logits
        return logits, attn_weights
