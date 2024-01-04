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
"""Registry utilities."""

import collections
import functools


class Registry(object):
    """Registry class."""

    def __init__(self, name):
        self.name = name
        self.registry = collections.OrderedDict()

    def has(self, key):
        return key in self.registry

    def register(self, name, func=None, **kwargs):
        def decorated(inner_function):
            for key in name if isinstance(name, (tuple, list)) else [name]:
                self.registry[key] = functools.partial(inner_function, **kwargs)
            return inner_function

        if func is not None:
            return decorated(func)
        return decorated

    def get(self, name, default=None):
        if name is None:
            return None
        if not self.has(name):
            if default is not None:
                return default
            raise KeyError("`%s` is not registered in <%s>." % (name, self.name))
        return self.registry[name]

    def try_get(self, name):
        if self.has(name):
            return self.get(name)
        return None
