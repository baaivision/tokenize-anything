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
"""Timing functions."""

import contextlib
import time


class Timer(object):
    """Simple timer."""

    def __init__(self):
        self.total_time = 0.0
        self.calls = 0
        self.start_time = 0.0
        self.diff = 0.0
        self.average_time = 0.0

    def add_diff(self, diff, n=1, average=True):
        self.total_time += diff
        self.calls += n
        self.average_time = self.total_time / self.calls
        return self.average_time if average else self.diff

    @contextlib.contextmanager
    def tic_and_toc(self, n=1, average=True):
        try:
            yield self.tic()
        finally:
            self.toc(n, average)

    def tic(self):
        self.start_time = time.time()
        return self

    def toc(self, n=1, average=True):
        self.diff = time.time() - self.start_time
        return self.add_diff(self.diff, n, average)
