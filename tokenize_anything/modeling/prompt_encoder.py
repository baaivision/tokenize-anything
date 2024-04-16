# ------------------------------------------------------------------------
# Copyright (c) 2023-present, BAAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
"""Prompt encoder."""

import torch
from torch import nn


class PromptEncoder(nn.Module):
    """Module to encode geometric prompts."""

    def __init__(self, embed_dim, image_size):
        super(PromptEncoder, self).__init__()
        self.point_embed = nn.Embedding(5, embed_dim)  # [bg, fg, lt, rb, pad]
        self.corner_labels = torch.tensor([[2, 3]], dtype=torch.int64)
        self.register_buffer("coord_matrix", torch.randn((2, embed_dim // 2)))
        self.img_pos, self.img_size = None, [image_size] * 2

    def as_tensor(self, input):
        """Convert input into a tensor."""
        return torch.as_tensor(input, device=self.coord_matrix.device)

    def to_points(self, points=None, boxes=None):
        """Convert points or boxes to point prompts."""
        if points is not None:
            if isinstance(points, (tuple, list)):
                coords, labels = points
            else:
                coords, labels = points[:, :, :2], points[:, :, 2]
            coords = coords.__add__(0.5).__itruediv__(self.img_size[::-1])
            coords = self.as_tensor(coords.clip(0, 1).astype("float32"))
            labels = self.as_tensor(labels.astype("int64"))
            return coords, labels
        if boxes is not None:
            coords = boxes.reshape((-1, 2, 2))
            coords = coords.__add__(0.5).__itruediv__(self.img_size[::-1])
            coords = self.as_tensor(coords.clip(0, 1).astype("float32"))
            labels = self.as_tensor(self.corner_labels)
            return coords, labels
        return None

    def encode_coords(self, coords):
        """Return the embedding for given coords."""
        pi4, pi2 = 4 * 3.1415926, 2 * 3.1415926
        if self.coord_matrix.dtype != torch.float32:
            self.coord_matrix = self.coord_matrix.float()
        rad = coords.mul(pi4).sub_(pi2) @ self.coord_matrix
        dtype = self.point_embed.weight.dtype
        return torch.cat([rad.sin(), rad.cos()], dim=-1).to(dtype=dtype)

    def encode_points(self, coords, labels):
        """Return the embedding for given points."""
        embed = self.encode_coords(coords)
        embed.mul_(labels.ne(4).unsqueeze_(-1).float().to(dtype=embed.dtype))
        return embed.add_(self.point_embed(labels))

    def encode_grid(self, grid_size):
        """Return the embedding for a grid of specified size."""
        grid = torch.ones(*grid_size, dtype=torch.float32)
        y = grid.cumsum(dim=0).sub_(0.5).div_(grid_size[0])
        x = grid.cumsum(dim=1).sub_(0.5).div_(grid_size[1])
        coords = self.as_tensor(torch.stack([x, y], dim=-1))
        return self.encode_coords(coords)

    def forward(self, inputs):
        sparse_embeds = []
        if inputs.get("boxes", None) is not None:
            coords, labels = self.to_points(boxes=inputs["boxes"])
            sparse_embeds.append(self.encode_points(coords, labels))
        if inputs.get("points", None) is not None:
            coords, labels = self.to_points(points=inputs["points"])
            sparse_embeds.append(self.encode_points(coords, labels))
        if len(sparse_embeds) > 1:
            sparse_embeds = [torch.cat(sparse_embeds, dim=1)]
        elif len(sparse_embeds) == 0:
            raise ValueError("Excepted ``points`` or ``boxes`` prompts.")
        img_embed_size = torch.Size(inputs["img_embeds"].shape[2:-1])
        if self.img_pos is None or self.img_pos.shape[0] != img_embed_size.numel():
            self.img_pos = self.encode_grid(img_embed_size).flatten(0, 1)
        return {"sparse_embeds": sparse_embeds[0], "img_pos": self.img_pos}
