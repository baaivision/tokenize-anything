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
"""Loss layers."""

from torch import nn


def reduce_loss(loss, reduction="mean"):
    """Reduce the loss."""
    if reduction == "mean" or reduction == "sum":
        return getattr(loss, reduction)()
    if reduction == "batch_mean":
        return loss.sum().mul_(1.0 / loss.size(0))
    return loss


class BinaryFocalLoss(nn.Module):
    """Binary focal loss."""

    def __init__(self, alpha=0.25, reduction="none"):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, input, target):
        alpha, p = self.alpha, input.sigmoid()
        neg_alpha, neg_target = 1.0 - alpha, 1.0 - target
        alpha_weight = target.mul(alpha).add_(neg_target.mul(neg_alpha))
        focal_weight = (1.0 - p).mul_(target).add_(p.mul(neg_target)).square()
        loss = nn.functional.binary_cross_entropy_with_logits(input, target, reduction="none")
        return reduce_loss(loss * focal_weight.mul_(alpha_weight), self.reduction)


class BinaryDiceLoss(nn.Module):
    """Binary dice loss."""

    def __init__(self, eps=1.0, reduction="none"):
        super(BinaryDiceLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, input, target):
        input = input.sigmoid()
        num = input.mul(target).sum(-1).mul_(2).add_(self.eps)
        den = input.add(target).sum(-1).add_(self.eps)
        return reduce_loss(1.0 - num / den, self.reduction)


class CrossEntropyLoss(nn.Module):
    """Cross entropy loss with label smoothing."""

    def __init__(self, epsilon=0, reduction="none"):
        super(CrossEntropyLoss, self).__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward_dense(self, input, target):
        dim, target = input.shape[-1], target.squeeze_()
        x = nn.functional.log_softmax(input, dim=-1)
        y = nn.functional.one_hot(target, dim).float()
        x = x.permute([0, x.dim() - 1] + list(range(x.dim()))[1:-1]) if x.dim() > 2 else x
        y = y.permute([0, y.dim() - 1] + list(range(y.dim()))[1:-1]) if y.dim() > 2 else y
        loss = nn.functional.cross_entropy(x, y, reduction="none", label_smoothing=self.epsilon)
        return reduce_loss(loss, self.reduction)

    def forward(self, input, target):
        if self.epsilon > 0:
            return self.forward_dense(input, target)
        return nn.functional.cross_entropy(input, target, reduction=self.reduction)
