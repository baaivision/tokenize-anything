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
"""Concet projector."""

import pickle

import numpy as np
import torch
from torch import nn


class ConceptProjector(nn.Module):
    """Encode and decode concept using CLIP."""

    def __init__(self, src_weights=None, tgt_weights=None):
        super(ConceptProjector, self).__init__()
        self.reset_weights(src_weights, tgt_weights)

    def reset_weights(self, src_weights=None, tgt_weights=None):
        """Reset the normalized projection weights."""
        if src_weights is not None:
            with open(src_weights, "rb") as f:
                self.src_weights, self.concepts = pickle.load(f)
                self.src_weights = torch.from_numpy(self.src_weights)
                self.concepts = np.array(self.concepts)
        if tgt_weights is not None:
            with open(tgt_weights, "rb") as f:
                self.tgt_weights, self.concepts = pickle.load(f)
                self.tgt_weights = torch.from_numpy(self.tgt_weights)
                self.concepts = np.array(self.concepts)

    @staticmethod
    def maybe_convert(embeds, proj):
        """Convert inputs for safe projection."""
        if embeds.dtype != torch.float32:
            embeds = embeds.float()
        if embeds.device != proj.device:
            proj = proj.to(device=embeds.device)
        return embeds, proj

    def encode_src(self, src_embeds):
        """Encode source visual embedding via concept projection."""
        src_embeds, self.src_weights = self.maybe_convert(src_embeds, self.src_weights)
        logits = nn.functional.normalize(src_embeds, dim=-1) @ self.src_weights
        return nn.functional.log_softmax(logits, dim=-1)

    def encode_tgt(self, tgt_embeds):
        """Encode target visual embedding via concept projection."""
        tgt_embeds, self.tgt_weights = self.maybe_convert(tgt_embeds, self.tgt_weights)
        logits = nn.functional.normalize(tgt_embeds, dim=-1) @ self.tgt_weights
        return nn.functional.log_softmax(logits, dim=-1)

    def decode(self, src_embeds, k=1, return_index=False, return_prob=False):
        """Return the top-k concepts of source visual embedding."""
        src_embeds, self.src_weights = self.maybe_convert(src_embeds, self.src_weights)
        logits = nn.functional.normalize(src_embeds, dim=-1) @ self.src_weights
        probs = nn.functional.softmax(logits, dim=-1)
        if return_prob:
            return probs.cpu().numpy()
        score, index = [x.cpu().numpy() for x in probs.topk(k, -1)]
        return (index if return_index else self.concepts[index]), score
