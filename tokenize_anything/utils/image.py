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
"""Image utilities."""

import numpy as np
import PIL.Image


def im_resize(img, size=None, scale=None, mode="linear"):
    """Resize image by the scale or size."""
    if size is None:
        if not isinstance(scale, (tuple, list)):
            scale = (scale, scale)
        h, w = img.shape[:2]
        size = int(h * scale[0] + 0.5), int(w * scale[1] + 0.5)
    else:
        if not isinstance(size, (tuple, list)):
            size = (size, size)
    resize_modes = {"linear": PIL.Image.BILINEAR}
    img = PIL.Image.fromarray(img)
    return np.array(img.resize(size[::-1], resize_modes[mode]))


def im_rescale(img, scales, max_size=0):
    """Rescale image to match the detecting scales."""
    im_shape = img.shape
    img_list, img_scales = [], []
    size_min = np.min(im_shape[:2])
    size_max = np.max(im_shape[:2])
    for target_size in scales:
        im_scale = float(target_size) / float(size_min)
        target_size_max = max_size if max_size > 0 else target_size
        if np.round(im_scale * size_max) > target_size_max:
            im_scale = float(target_size_max) / float(size_max)
        img_list.append(im_resize(img, scale=im_scale))
        img_scales.append((im_scale, im_scale))
    return img_list, img_scales


def im_vstack(arrays, fill_value=None, dtype=None, size=None, align=None):
    """Stack image arrays in sequence vertically."""
    if fill_value is None:
        return np.vstack(arrays)
    # Compute the max stack shape.
    max_shape = np.max(np.stack([arr.shape for arr in arrays]), 0)
    if size is not None and min(size) > 0:
        max_shape[: len(size)] = size
    if align is not None and min(align) > 0:
        align_size = np.ceil(max_shape[: len(align)] / align)
        max_shape[: len(align)] = align_size.astype("int64") * align
    # Fill output with the given value.
    output_dtype = dtype or arrays[0].dtype
    output_shape = [len(arrays)] + list(max_shape)
    output = np.empty(output_shape, output_dtype)
    output[:] = fill_value
    # Copy arrays.
    for i, arr in enumerate(arrays):
        copy_slices = (slice(0, d) for d in arr.shape)
        output[(i,) + tuple(copy_slices)] = arr
    return output
