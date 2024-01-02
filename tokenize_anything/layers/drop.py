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
"""Drop regularization layers."""

from torch import nn


class DropPath(nn.Module):
    """Set examples to zero randomly."""

    def __init__(self, p=0.1, inplace=False):
        super(DropPath, self).__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, input):
        if not self.training or self.p <= 0:
            return input
        keep_p = 1 - self.p
        shape = (input.shape[0],) + (1,) * (input.ndim - 1)
        scale = input.new_empty(shape).bernoulli_(keep_p).div_(keep_p)
        return input.mul_(scale) if self.inplace else input.mul(scale)

    def extra_repr(self):
        inplace_str = ", inplace" if self.inplace else ""
        return "p={}{}".format(self.p, inplace_str)
