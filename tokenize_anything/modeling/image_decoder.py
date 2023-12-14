# ------------------------------------------------------------------------
# Copyright (c) 2023-present, BAAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
"""Image decoder."""

try:
    from flash_attn import flash_attn_func
except ImportError:
    flash_attn_func = None

import torch
from torch import nn


class TransposedLayerNorm(nn.LayerNorm):
    """LayerNorm with pre-transposed spatial axes."""

    def forward(self, input):
        return super().forward(input.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class MLP(nn.Module):
    """Two layers MLP."""

    def __init__(self, dim, mlp_dim, activation_type="ReLU"):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, dim)
        self.activation = getattr(nn, activation_type)()
        self.activation.inplace = True

    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))


class Attention(nn.Module):
    """Multi-head attention."""

    def __init__(self, dim=256, num_heads=8, attn_ratio=1):
        super(Attention, self).__init__()
        qkv_dim = int(dim * attn_ratio)
        self.num_heads = num_heads
        self.head_dim = qkv_dim // num_heads
        self.q_proj = nn.Linear(dim, qkv_dim)
        self.k_proj = nn.Linear(dim, qkv_dim)
        self.v_proj = nn.Linear(dim, qkv_dim)
        self.proj = nn.Linear(qkv_dim, dim)
        self.scale = self.head_dim**-0.5

    def forward(self, q, k, v):
        q = self.q_proj(q).view((-1, q.size(1), self.num_heads, self.head_dim))
        k = self.k_proj(k).view((-1, k.size(1), self.num_heads, self.head_dim))
        v = self.v_proj(v).view((-1, v.size(1), self.num_heads, self.head_dim))
        o = flash_attn_func(q, k, v, softmax_scale=self.scale)
        return self.proj(o.flatten(2))


class Block(nn.Module):
    """Transformer block."""

    def __init__(
        self,
        dim=256,
        num_heads=8,
        attn_ratio=0.5,
        mlp_dim=2048,
        dropout=0.1,
        activation_type="ReLU",
        skip_first_query_pos=False,
    ):
        super(Block, self).__init__()
        self.self_attn = Attention(dim, num_heads)
        self.norm1 = nn.LayerNorm(dim)
        self.cross_attn_token_to_image = Attention(dim, num_heads, attn_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_dim, activation_type)
        self.norm3 = nn.LayerNorm(dim)
        self.cross_attn_image_to_token = Attention(dim, num_heads, attn_ratio)
        self.norm4 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.skip_first_query_pos = skip_first_query_pos

    def forward(self, query, key, query_pos, key_pos):
        if self.skip_first_query_pos:
            query = self.norm1(self.self_attn(query, query, query))
        else:
            q = query + query_pos
            query = self.norm1(self.dropout(self.self_attn(q, q, query)).add_(query))
        q, k = query + query_pos, key + key_pos
        query = self.norm2(self.dropout(self.cross_attn_token_to_image(q, k, key)).add_(query))
        query = self.norm3(self.dropout(self.mlp(query)).add_(query))
        q = query + query_pos
        key = self.norm4(self.cross_attn_image_to_token(k, q, query).add_(key))
        return query, key


class Transformer(nn.Module):
    """Two-way transformer decoder."""

    def __init__(
        self,
        embed_dim=256,
        num_heads=8,
        attn_ratio=0.5,
        mlp_dim=2048,
        dropout=0.1,
        activation_type="ReLU",
        depth=2,
    ):
        super(Transformer, self).__init__()
        self.blocks = nn.ModuleList(
            Block(
                embed_dim,
                num_heads,
                attn_ratio=attn_ratio,
                mlp_dim=mlp_dim,
                dropout=dropout,
                activation_type=activation_type,
                skip_first_query_pos=i == 0,
            )
            for i in range(depth)
        )
        self.final_attn_token_to_image = Attention(embed_dim, num_heads, attn_ratio)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout, inplace=True)

    def forward(self, query, key, query_pos, key_pos):
        for blk in self.blocks:
            query, key = blk(query, key, query_pos, key_pos)
        q, k = query + query_pos, key + key_pos
        query = self.dropout(self.final_attn_token_to_image(q, k, key)).add_(query)
        query = self.norm(query)
        return query, key


class Predictor(nn.Module):
    """MLP predictor."""

    def __init__(self, in_dim, out_dim, mlp_dim=None, depth=3):
        super(Predictor, self).__init__()
        mlp_dims = [mlp_dim or in_dim] * (depth - 1)
        in_dims, out_dims = [in_dim] + mlp_dims, mlp_dims + [out_dim]
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip(in_dims, out_dims))

    def forward(self, x):
        for fc in self.layers[:-1]:
            x = nn.functional.relu(fc(x), inplace=True)
        return self.layers[-1](x)


class ImageDecoder(nn.Module):
    """Module to decode region tokens and masks."""

    def __init__(self, depth, embed_dim, num_heads, num_mask_tokens=4, sem_embed_dim=1024):
        super(ImageDecoder, self).__init__()
        self.embed_dim = embed_dim
        self.num_mask_tokens = num_mask_tokens
        self.transformer = Transformer(embed_dim, num_heads=num_heads, depth=depth)
        self.iou_token = nn.Embedding(1, embed_dim)
        self.sem_tokens = nn.Embedding(self.num_mask_tokens, embed_dim)
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, embed_dim)
        self.output_conv = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 4, 2, 2),
            TransposedLayerNorm(embed_dim // 4),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim // 4, embed_dim // 8, 2, 2),
            nn.GELU(),
        )
        self.mask_pred = nn.ModuleList(
            Predictor(embed_dim, embed_dim // 8) for _ in range(num_mask_tokens)
        )
        self.iou_pred = Predictor(embed_dim, self.num_mask_tokens)
        self.sem_pred = Predictor(embed_dim, sem_embed_dim, 1024)

    def get_outputs(self, inputs):
        img_embeds = inputs["img_embeds"]
        sparse_embeds = inputs["sparse_embeds"]
        ims_per_batch = img_embeds.size(0)
        prompts_per_batch = sparse_embeds.size(0)
        img_embed_size = img_embeds.shape[2:-1]
        # Prepare query.
        tokens = [self.sem_tokens.weight, self.iou_token.weight, self.mask_tokens.weight]
        query = torch.cat(tokens).unsqueeze_(0).expand(prompts_per_batch, -1, -1)
        query = torch.cat((query, sparse_embeds), dim=1)
        num_tokens = query.shape[1] - sparse_embeds.shape[1]
        # Prepare key.
        key = img_embeds.expand(-1, prompts_per_batch // ims_per_batch, -1, -1, -1)
        key = key.flatten(0, 1).flatten(1, 2)
        # Decode.
        query, key = self.transformer(query, key, query, inputs["img_pos"])
        # Upscale key.
        key = key.transpose(1, 2).view((-1, self.embed_dim) + img_embed_size)
        output_masks = self.output_conv(key).flatten(2)
        # Unpack query.
        tokens = query[:, :num_tokens].unbind(dim=1)
        iou_tokens = tokens[num_tokens - self.num_mask_tokens - 1]
        mask_tokens = tokens[num_tokens - self.num_mask_tokens :]
        sem_tokens = tokens[: self.num_mask_tokens]
        # Predict.
        mask_pred = [f(x) for f, x in zip(self.mask_pred, mask_tokens)]
        mask_pred = torch.stack(mask_pred, dim=1) @ output_masks
        mask_pred_size = list(4 * embed_size for embed_size in img_embed_size)
        mask_pred = mask_pred.view([-1, self.num_mask_tokens] + mask_pred_size)
        outputs = {"iou_pred": self.iou_pred(iou_tokens), "mask_pred": mask_pred}
        outputs["sem_tokens"] = torch.stack(sem_tokens, dim=1)
        outputs["sem_embeds"] = self.sem_pred(outputs["sem_tokens"])
        return outputs

    def forward(self, inputs):
        outputs = self.get_outputs(inputs)
        outputs["iou_pred"] = outputs["iou_pred"].float()
        return outputs
