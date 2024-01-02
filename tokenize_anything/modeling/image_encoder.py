# Copyright (c) 2023-present, BAAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################
"""Image encoder."""

import torch
from torch import nn

from tokenize_anything import layers


def space_to_depth(input, block_size):
    """Rearrange blocks of spatial data into depth."""
    if input.dim() == 3:
        hXw, c = input.size()[1:]
        h = w = int(hXw**0.5)
    else:
        h, w, c = input.size()[1:]
    h1, w1 = h // block_size, w // block_size
    c1 = (block_size**2) * c
    input = input.reshape((-1, h1, block_size, w1, block_size, c))
    return input.permute(0, 1, 3, 2, 4, 5).reshape((-1, h1, w1, c1))


def depth_to_space(input, block_size):
    """Rearrange blocks of depth data into spatial."""
    h1, w1, c1 = input.size()[1:]
    h, w = h1 * block_size, w1 * block_size
    c = c1 // (block_size**2)
    input = input.reshape((-1, h1, w1, block_size, block_size, c))
    return input.permute(0, 1, 3, 2, 4, 5).reshape((-1, h, w, c))


class MLP(nn.Module):
    """Two layers MLP."""

    def __init__(self, dim, mlp_ratio=4):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(dim, int(dim * mlp_ratio))
        self.fc2 = nn.Linear(int(dim * mlp_ratio), dim)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))


class Attention(nn.Module):
    """Multihead attention."""

    def __init__(self, dim, num_heads, qkv_bias=True):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.rel_pos_embed = nn.Identity()

    def forward(self, x):
        qkv_shape = (-1, x.size(1), 3, self.num_heads, self.head_dim)
        qkv = self.qkv(x).reshape(qkv_shape).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0)
        attn = q @ k.transpose(-2, -1).mul(self.scale)
        attn = self.rel_pos_embed(attn)
        o = nn.functional.softmax(attn, dim=-1) @ v
        return self.proj(o.transpose(1, 2).flatten(2))


class Block(nn.Module):
    """Transformer block."""

    def __init__(self, dim, num_heads, mlp_ratio=4, qkv_bias=True):
        super(Block, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, qkv_bias=qkv_bias)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio)
        self.drop_path = layers.DropPath(0.1, inplace=True)

    def forward(self, x):
        x = self.drop_path(self.attn(self.norm1(x))).add_(x)
        return self.drop_path(self.mlp(self.norm2(x))).add_(x)


class Bottleneck(nn.Module):
    """The bottleneck block."""

    def __init__(self, dim, expansion=2, width=None):
        super(Bottleneck, self).__init__()
        width = width or dim // expansion
        self.conv1 = nn.Conv2d(dim, width, 1, bias=False)
        self.norm1 = nn.SyncBatchNorm(width)
        self.conv2 = nn.Conv2d(width, width, 3, padding=1, bias=False)
        self.norm2 = nn.SyncBatchNorm(width)
        self.conv3 = nn.Conv2d(width, dim, 1, bias=False)
        self.norm3 = nn.SyncBatchNorm(dim)
        self.activation = nn.GELU()

    def forward(self, x):
        shortcut = x
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.activation(self.norm2(self.conv2(x)))
        return self.norm3(self.conv3(x)).add_(shortcut)


class PatchEmbed(nn.Module):
    """Patch embedding layer."""

    def __init__(self, dim=768, patch_size=16, bias=True):
        super(PatchEmbed, self).__init__()
        self.proj = nn.Conv2d(3, dim, patch_size, patch_size, bias=bias)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class PosEmbed(nn.Module):
    """Position embedding layer."""

    def __init__(self, dim, num_patches):
        super(PosEmbed, self).__init__()
        self.dim = dim
        self.num_patches = num_patches
        self.weight = nn.Parameter(torch.zeros(num_patches, dim))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        return x.add_(self.weight)


class RelPosEmbed(nn.Module):
    """Relative position embedding layer."""

    def __init__(self, num_heads, size):
        super(RelPosEmbed, self).__init__()
        self.register_buffer("index", self.get_index(size))
        self.weight = nn.Parameter(torch.zeros(num_heads, (2 * size - 1) ** 2))

    @staticmethod
    def get_index(size):
        """Return the relative index."""
        grid = torch.arange(size)
        grid = torch.stack(torch.meshgrid(grid, grid, indexing="ij")).reshape((2, -1))
        coords = grid[:, :, None] - grid[:, None, :] + (size - 1)
        coords[0] *= 2 * size - 1
        return coords.sum(0)

    def get_bias(self):
        return self.weight[:, self.index]

    def forward(self, x):
        return x.add_(self.get_bias())


class SimpleFeaturePyramid(nn.Module):
    """Module to create pyramid features."""

    def __init__(self, embed_dim, out_dim, patch_size=16, min_lvl=4, max_lvl=4):
        super(SimpleFeaturePyramid, self).__init__()
        self.min_lvl, self.max_lvl = min_lvl, max_lvl
        self.input_conv = nn.ModuleList()
        self.lateral_conv = nn.ModuleList()
        self.output_conv = nn.ModuleList()
        patch_lvl = dict((2**i, i) for i in range(6))[patch_size]
        for lvl in [min(i + 2, self.max_lvl) for i in range(4)]:
            if lvl == patch_lvl or lvl < self.min_lvl:
                self.input_conv += [nn.Identity()]
            elif lvl < patch_lvl:
                stride, layers = 2 ** (patch_lvl - lvl), []
                while stride > 1:
                    layers += [nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)]
                    layers += [nn.SyncBatchNorm(embed_dim), nn.GELU()] if stride > 2 else []
                    stride /= 2
                self.input_conv.append(nn.Sequential(*layers))
            elif lvl > patch_lvl:
                stride = 2 ** (lvl - patch_lvl)
                self.input_conv += [nn.MaxPool2d(stride, stride)]
        for _ in range(min_lvl, max_lvl + 1):
            self.lateral_conv.append(
                nn.Sequential(
                    nn.Conv2d(embed_dim, out_dim, kernel_size=1, bias=False),
                    nn.SyncBatchNorm(out_dim),
                )
            )
            self.output_conv.append(
                nn.Sequential(
                    nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=False),
                    nn.SyncBatchNorm(out_dim),
                )
            )

    def forward(self, inputs):
        inputs = inputs + [inputs[-1]] * (4 - len(inputs))
        inputs = [conv(x) for conv, x in zip(self.input_conv, inputs)]
        features = inputs[self.min_lvl - 1 : self.max_lvl]
        laterals = [conv(x) for conv, x in zip(self.lateral_conv, features)]
        return [conv(x) for conv, x in zip(self.output_conv, laterals)]


class ImageEncoderViT(nn.Module):
    """ViT image encoder."""

    def __init__(
        self,
        depth,
        embed_dim,
        num_heads,
        mlp_ratio=4,
        patch_size=16,
        window_size=16,
        image_size=1024,
        out_dim=256,
    ):
        super(ImageEncoderViT, self).__init__()
        self.embed_dim = embed_dim
        self.image_size = image_size
        self.window_size = window_size or image_size // patch_size
        self.patch_embed = PatchEmbed(embed_dim, patch_size)
        self.pos_embed = PosEmbed(embed_dim, (image_size // patch_size) ** 2)
        self.blocks = nn.ModuleList(Block(embed_dim, num_heads, mlp_ratio) for _ in range(depth))
        for blk in self.blocks:
            blk.attn.rel_pos_embed = RelPosEmbed(num_heads, self.window_size)
        self.norm = nn.LayerNorm(embed_dim)
        self.cross_conv = nn.ModuleList(Bottleneck(embed_dim) for _ in range(4))
        self.neck = SimpleFeaturePyramid(embed_dim, out_dim, patch_size)
        self.cross_indices = list(range(depth // 4 - 1, depth, depth // 4))

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.pos_embed(x)
        x = space_to_depth(x, self.window_size)
        wmsa_shape = (-1,) + x.shape[1:]
        msa_shape = (-1, self.window_size**2, self.embed_dim)
        x = x.reshape(msa_shape)
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.cross_indices or i == len(self.blocks) - 1:
                x = self.norm(x) if i == len(self.blocks) - 1 else x
                x = depth_to_space(x.reshape(wmsa_shape), self.window_size)
                x = x.permute(0, 3, 1, 2)
            if i in self.cross_indices:
                x = self.cross_conv[self.cross_indices.index(i)](x)
            if i in self.cross_indices and i < len(self.blocks) - 1:
                x = x.permute(0, 2, 3, 1)
                x = space_to_depth(x, self.window_size).reshape(msa_shape)
        return self.neck([x])
