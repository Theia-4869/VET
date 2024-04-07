#!/usr/bin/env python3
"""
vit with ssf
"""
import copy
import numpy as np
import torch
import torch.nn as nn

from torch.nn import Parameter, Linear, LayerNorm
from ..vit_backbones.vit import *
from ...utils import logging
logger = logging.get_logger("PEFT")


class SSFAttention(Attention):
    def __init__(self, config, ssf_config, vis):
        super(SSFAttention, self).__init__(config, vis)

        self.ssf_config = ssf_config

        self.query_ssf_scale = Parameter(torch.ones(self.all_head_size))
        self.query_ssf_shift = Parameter(torch.zeros(self.all_head_size))
        self.key_ssf_scale = Parameter(torch.ones(self.all_head_size))
        self.key_ssf_shift = Parameter(torch.zeros(self.all_head_size))
        self.value_ssf_scale = Parameter(torch.ones(self.all_head_size))
        self.value_ssf_shift = Parameter(torch.zeros(self.all_head_size))
        self.out_ssf_scale = Parameter(torch.ones(config.hidden_size))
        self.out_ssf_shift = Parameter(torch.zeros(config.hidden_size))

        nn.init.normal_(self.query_ssf_scale, mean=1.0, std=0.02)
        nn.init.normal_(self.query_ssf_shift, std=0.02)
        nn.init.normal_(self.key_ssf_scale, mean=1.0, std=0.02)
        nn.init.normal_(self.key_ssf_shift, std=0.02)
        nn.init.normal_(self.value_ssf_scale, mean=1.0, std=0.02)
        nn.init.normal_(self.value_ssf_shift, std=0.02)
        nn.init.normal_(self.out_ssf_scale, mean=1.0, std=0.02)
        nn.init.normal_(self.out_ssf_shift, std=0.02)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states) # B, num_patches, head_size*num_head
        mixed_query_layer = mixed_query_layer * self.query_ssf_scale + self.query_ssf_shift
        mixed_key_layer = self.key(hidden_states)
        mixed_key_layer = mixed_key_layer * self.key_ssf_scale + self.key_ssf_shift
        mixed_value_layer = self.value(hidden_states)
        mixed_value_layer = mixed_value_layer * self.value_ssf_scale + self.value_ssf_shift

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
        attention_output = attention_output * self.out_ssf_scale + self.out_ssf_shift
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class SSFMlp(Mlp):
    def __init__(self, config, ssf_config):
        super(SSFMlp, self).__init__(config)

        self.ssf_config = ssf_config

        self.fc1_ssf_scale = Parameter(torch.ones(config.transformer["mlp_dim"]))
        self.fc1_ssf_shift = Parameter(torch.zeros(config.transformer["mlp_dim"]))
        self.fc2_ssf_scale = Parameter(torch.ones(config.hidden_size))
        self.fc2_ssf_shift = Parameter(torch.zeros(config.hidden_size))

        nn.init.normal_(self.fc1_ssf_scale, mean=1.0, std=0.02)
        nn.init.normal_(self.fc1_ssf_shift, std=0.02)
        nn.init.normal_(self.fc2_ssf_scale, mean=1.0, std=0.02)
        nn.init.normal_(self.fc2_ssf_shift, std=0.02)

    def forward(self, x):
        x = self.fc1(x)
        x = x * self.fc1_ssf_scale + self.fc1_ssf_shift
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = x * self.fc2_ssf_scale + self.fc2_ssf_shift
        x = self.dropout(x)
        return x


class SSFBlock(Block):
    def __init__(self, config, ssf_config, vis):
        super(SSFBlock, self).__init__(config, vis)

        self.ssf_config = ssf_config

        self.attention_norm_ssf_scale = Parameter(torch.ones(config.hidden_size))
        self.attention_norm_ssf_shift = Parameter(torch.zeros(config.hidden_size))
        self.attn = SSFAttention(config, ssf_config, vis)
        self.ffn_norm_ssf_scale = Parameter(torch.ones(config.hidden_size))
        self.ffn_norm_ssf_shift = Parameter(torch.zeros(config.hidden_size))
        self.ffn = SSFMlp(config, ssf_config)

        nn.init.normal_(self.attention_norm_ssf_scale, mean=1.0, std=0.02)
        nn.init.normal_(self.attention_norm_ssf_shift, std=0.02)
        nn.init.normal_(self.ffn_norm_ssf_scale, mean=1.0, std=0.02)
        nn.init.normal_(self.ffn_norm_ssf_shift, std=0.02)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x = x * self.attention_norm_ssf_scale + self.attention_norm_ssf_shift
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = x * self.ffn_norm_ssf_scale + self.ffn_norm_ssf_shift
        x = self.ffn(x)
        x = x + h 

        return x, weights


class SSFEmbeddings(Embeddings):
    def __init__(self, config, ssf_cfg, img_size=224):
        super(SSFEmbeddings, self).__init__(config, img_size)

        self.ssf_config = ssf_cfg
        
        self.patch_embeddings_ssf_scale = Parameter(torch.ones(config.hidden_size))
        self.patch_embeddings_ssf_shift = Parameter(torch.zeros(config.hidden_size))

        nn.init.normal_(self.patch_embeddings_ssf_scale, mean=1.0, std=0.02)
        nn.init.normal_(self.patch_embeddings_ssf_shift, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)

        if self.hybrid:
            x = self.hybrid_model(x)
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        x = x * self.patch_embeddings_ssf_scale + self.patch_embeddings_ssf_shift
        x = torch.cat((cls_tokens, x), dim=1)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class SSFEncoder(Encoder):
    def __init__(self, config, ssf_cfg, vis):
        super(SSFEncoder, self).__init__(config, vis)

        self.ssf_config = ssf_cfg

        self.layer = nn.ModuleList()
        self.num_layers = config.transformer["num_layers"]
        for _ in range(self.num_layers):
            layer = SSFBlock(config, ssf_cfg, vis)
            self.layer.append(copy.deepcopy(layer))

        self.encoder_norm_ssf_scale = Parameter(torch.ones(config.hidden_size))
        self.encoder_norm_ssf_shift = Parameter(torch.zeros(config.hidden_size))

        nn.init.normal_(self.encoder_norm_ssf_scale, mean=1.0, std=0.02)
        nn.init.normal_(self.encoder_norm_ssf_shift, std=0.02)

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        encoded = encoded * self.encoder_norm_ssf_scale + self.encoder_norm_ssf_shift
        return encoded, attn_weights


class SSFTransformer(Transformer):
    def __init__(self, config, ssf_cfg, img_size, vis):
        super(SSFTransformer, self).__init__(config, img_size, vis)

        self.vit_config = config
        self.ssf_cfg = ssf_cfg

        self.embeddings = SSFEmbeddings(config, ssf_cfg, img_size=img_size)
        self.encoder = SSFEncoder(config, ssf_cfg, vis)

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)

        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights


class SSFVisionTransformer(VisionTransformer):
    def __init__(
        self, model_type, img_size=224, num_classes=21843, ssf_cfg=None, vis=False
    ):
        super(SSFVisionTransformer, self).__init__(model_type, img_size, num_classes, vis)

        vit_cfg = CONFIGS[model_type]
        if ssf_cfg is None:
            raise ValueError("ssf_cfg cannot be None if using SSFVisionTransformer")

        self.ssf_cfg = ssf_cfg
        self.transformer = SSFTransformer(vit_cfg, ssf_cfg, img_size, vis)

    def forward(self, x, vis=False):
        x, attn_weights = self.transformer(x)
        logits = self.head(x[:, 0])

        if not vis:
            return logits
        return logits, attn_weights
