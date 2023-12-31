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
"""Generate visual prompts."""

import collections

import numpy as np
import numpy.random as npr


class VisualPrompter(object):
    """Generate visual prompts."""

    def __init__(self, image_size=1024, max_points=9, num_experts=4, padding_index=4):
        super(VisualPrompter, self).__init__()
        self.num_stages = 2
        self.max_points = max_points
        self.point_weight = [1000] + [0] * (num_experts - 1)
        self.image_size = image_size if isinstance(image_size, (tuple, list)) else [image_size] * 2
        self.padding_index = padding_index
        self.coord_count = collections.defaultdict(int)
        self.coords = self.labels = self.boxes_turn = None
        self.stage_count = 0
        self.box_prob = 0.5

    @property
    def is_last_stage(self):
        return self.stage_count == self.num_stages - 1

    def add_point(self, index, gt_masks, error_masks=None, num=1):
        def sample(mask):
            ys, xs = np.nonzero(mask)
            if ys.shape[0] > 0:
                idx = npr.choice(ys.shape[0], size=(num,), replace=num > ys.shape[0])
                return xs[idx], ys[idx]
            return [-0.5] * num, [-0.5] * num

        labels = [self.padding_index] * num
        if error_masks is not None:  # FP or FN point.
            xs, ys = sample(error_masks[index])
            labels = gt_masks[index, ys, xs] if ys[0] >= 0 else labels
        if labels[0] == self.padding_index:  # GT point.
            xs, ys = sample(gt_masks[index])
            labels = [1] * num if ys[0] >= 0 else labels
        xs = (np.array(xs, "float32") + 0.5) * (self.image_size[1] / gt_masks.shape[2]) - 0.5
        ys = (np.array(ys, "float32") + 0.5) * (self.image_size[0] / gt_masks.shape[1]) - 0.5
        slice_index = slice(self.coord_count[index], self.coord_count[index] + num)
        self.coords[index, slice_index] = np.vstack([xs, ys]).T
        self.labels[index, slice_index] = labels
        self.coord_count[index] += num

    def add_box(self, index, gt_boxes):
        x1, y1, x2, y2 = gt_boxes[index, :4]
        dx1, dx2 = np.clip(npr.normal(0.0, 0.1 * (x2 - x1), (2,)), -20, 20)
        dy1, dy2 = np.clip(npr.normal(0.0, 0.1 * (y2 - y1), (2,)), -20, 20)
        x1, y1 = x1 + np.minimum(dx1, 0), y1 + np.minimum(dy1, 0)
        x2, y2 = x2 + np.maximum(dx2, 0), y2 + np.maximum(dy2, 0)
        self.coords[index, self.coord_count[index]] = (x1, y1)
        self.coords[index, self.coord_count[index] + 1] = (x2, y2)
        self.labels[index, self.coord_count[index]] = 2
        self.labels[index, self.coord_count[index] + 1] = 3
        self.coord_count[index] += 2

    def reset(self, num):
        self.stage_count = 0
        self.coord_count.clear()
        self.coords = np.full((num, self.max_points + 1, 2), -0.5, "float32")
        self.labels = np.full((num, self.max_points + 1), self.padding_index, "int64")
        self.boxes_turn = npr.rand(num) < self.box_prob

    def get_prompts(self, gt_boxes, gt_masks=None, masks=None):
        num = gt_boxes.shape[0]
        if self.stage_count == 0:
            self.reset(num)
        coords = labels = error_masks = None
        if masks is not None:
            masks = masks.reshape(gt_masks.shape)
            error_masks = (masks | gt_masks) ^ (masks & gt_masks)
        num_points = 1
        if self.stage_count > 0:
            num_points = npr.randint(1, self.max_points + 1 - self.stage_count)
        if self.stage_count == 0 and self.box_prob == 0:
            num_points = npr.randint(2, self.max_points + 1)
        for index in range(num):
            is_box = self.stage_count == 0 and self.boxes_turn[index]
            if gt_masks is None or is_box:
                self.add_box(index, gt_boxes)
            else:
                self.add_point(index, gt_masks, error_masks, num_points)
        coords = self.coords[:, : 1 + self.stage_count + num_points]
        labels = self.labels[:, : 1 + self.stage_count + num_points]
        scores = (self.boxes_turn[:, None] - 0.5) * self.point_weight
        return {"points": (coords, labels), "point_score": scores}
