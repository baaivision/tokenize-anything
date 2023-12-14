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
"""Mask utilities."""

import numpy as np
from pycocotools.mask import encode


def mask_to_box(mask):
    """Convert binary masks to boxes."""
    shape, (h, w) = mask.shape, mask.shape[-2:]
    masks = mask.reshape((-1, h, w)).astype("bool")
    in_height = np.max(masks, axis=-1)
    in_width = np.max(masks, axis=-2)
    in_height_coords = in_height * np.arange(h, dtype="int32")
    in_width_coords = in_width * np.arange(w, dtype="int32")
    bottom_edges = np.max(in_height_coords, axis=-1)
    top_edges = np.min(in_height_coords + h * (~in_height), axis=-1)
    right_edges = np.max(in_width_coords, axis=-1)
    left_edges = np.min(in_width_coords + w * (~in_width), axis=-1)
    is_empty = (right_edges < left_edges) | (bottom_edges < top_edges)
    boxes = np.stack([left_edges, top_edges, right_edges, bottom_edges], axis=-1)
    boxes = boxes.astype("float32") * ((~is_empty)[:, None])
    return boxes.reshape(*shape[:-2], 4) if len(shape) > 2 else boxes[0]


def encode_masks(masks):
    """Encode a set of masks to RLEs."""
    rles = encode(np.asfortranarray(masks))
    for rle in rles:
        rle["counts"] = rle["counts"].decode()
    return rles
