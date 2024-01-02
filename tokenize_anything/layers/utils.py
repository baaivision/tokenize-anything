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
"""Layer utilities."""

import cv2
import numpy as np
import torch


def set_dropout(module, dropout):
    """Initialize dropout."""
    for m in [m for m in module.modules() if isinstance(m, torch.nn.Dropout)]:
        m.p = dropout


def set_drop_path(blocks, drop_path):
    """Initialize drop path."""
    if not isinstance(blocks, torch.nn.ModuleList):
        blocks = getattr(blocks, "blocks", getattr(blocks, "layers", None))
    for i, blk in enumerate(blocks):
        for m in [m for m in blk.modules() if type(m).__name__ == "DropPath"]:
            m.p = i * drop_path / (len(blocks) - 1)


def resize_pos_embed(weight, out_len):
    """Resize position embedding weights."""
    out_h = out_w = int(out_len**0.5)
    h = w = int(weight.shape[0] ** 0.5)
    weight = weight.reshape((h, w, weight.shape[1])).astype("float32", copy=False)
    out_weight = [
        cv2.resize(x, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
        for x in np.split(weight, 4, axis=-1)
    ]
    return np.concatenate(out_weight, axis=-1).reshape((-1, weight.shape[-1]))
