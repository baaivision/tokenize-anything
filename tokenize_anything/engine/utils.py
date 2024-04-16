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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, esither express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
"""Engine utilities."""

import collections
import functools
import pickle

import torch
from torch import nn
import numpy as np

from tokenize_anything.utils import logging

GLOBAL_DDP_GROUP = None


def count_params(module, trainable=True, unit="M"):
    """Return the number of parameters."""
    counts = [v.size().numel() for v in module.parameters() if v.requires_grad or (not trainable)]
    return sum(counts) / {"M": 1e6, "B": 1e9}[unit]


def freeze_module(module, trainable=False):
    """Freeze parameters of given module."""
    module.eval() if not trainable else module.train()
    for param in module.parameters():
        param.requires_grad = trainable


def get_device(index):
    """Create the available device object."""
    if torch.cuda.is_available():
        return torch.device("cuda", index)
    for device_type in ("mps",):
        try:
            if getattr(torch.backends, device_type).is_available():
                return torch.device(device_type, index)
        except AttributeError:
            pass
    return torch.device("cpu")


def get_param_groups(model, layer_lr_decay=1.0):
    """Separate parameters into groups."""
    memo, groups, lr_scale_getter = set(), collections.OrderedDict(), None
    if layer_lr_decay < 1.0 and hasattr(model.image_encoder, "get_lr_scale"):
        lr_scale_getter = functools.partial(model.image_encoder.get_lr_scale, decay=layer_lr_decay)
    norm_types = (nn.BatchNorm2d, nn.GroupNorm, nn.SyncBatchNorm, nn.LayerNorm)
    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            if not param.requires_grad or param in memo:
                continue
            memo.add(param)
            attrs = collections.OrderedDict()
            if lr_scale_getter:
                attrs["lr_scale"] = lr_scale_getter(f"{module_name}.{param_name}")
            if hasattr(param, "lr_scale"):
                attrs["lr_scale"] = param.lr_scale
            if getattr(param, "no_weight_decay", False) or isinstance(module, norm_types):
                attrs["weight_decay"] = 0
            group_name = "/".join(["%s:%s" % (v[0], v[1]) for v in list(attrs.items())])
            groups[group_name] = groups.get(group_name, {**attrs, **{"params": []}})
            groups[group_name]["params"].append(param)
    return list(groups.values())


def load_weights(module, weights_file, prefix_removed="", strict=True):
    """Load a weights file."""
    if not weights_file:
        return
    if weights_file.endswith(".pkl"):
        with open(weights_file, "rb") as f:
            state_dict = pickle.load(f)
            for k, v in state_dict.items():
                state_dict[k] = torch.as_tensor(v)
    else:
        state_dict = torch.load(weights_file)
    if prefix_removed:
        new_state_dict = type(state_dict)()
        for k in list(state_dict.keys()):
            new_state_dict[k.replace(prefix_removed, "")] = state_dict.pop(k)
        state_dict = new_state_dict
    module.load_state_dict(state_dict, strict=strict)


def manual_seed(seed, device_and_seed=None):
    """Set the cpu and device random seed."""
    torch.manual_seed(seed)
    if device_and_seed is not None:
        device_index, device_seed = device_and_seed
        device_type = get_device(device_index).type
        np.random.seed(device_seed)
        if device_type in ("cuda", "mps"):
            getattr(torch, device_type).manual_seed(device_seed)


def synchronize_device(device):
    """Synchronize the computation of device."""
    if device.type in ("cuda", "mps"):
        getattr(torch, device.type).synchronize(device)


def create_ddp_group(cfg, ranks=None, devices=None):
    """Create group for data parallelism."""
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")
    world_rank = torch.distributed.get_rank()
    ranks = ranks if ranks else [i for i in range(cfg.NUM_GPUS)]
    logging.set_root(world_rank == ranks[0])
    devices = devices if devices else [i % 8 for i in range(len(ranks))]
    cfg.GPU_ID = devices[world_rank]
    torch.cuda.set_device(cfg.GPU_ID)
    global GLOBAL_DDP_GROUP
    GLOBAL_DDP_GROUP = torch.distributed.new_group(ranks)
    return GLOBAL_DDP_GROUP


def get_ddp_group():
    """Return the process group for data parallelism."""
    return GLOBAL_DDP_GROUP


def get_ddp_rank():
    """Return the rank in the data parallelism group."""
    ddp_group = get_ddp_group()
    if ddp_group is None:
        return 0
    return torch.distributed.get_rank(ddp_group)


def apply_ddp_group(module):
    """Apply data parallelism group for given module."""
    ddp_group = get_ddp_group()
    if ddp_group is None:
        return module
    return torch.nn.parallel.DistributedDataParallel(module, process_group=ddp_group)


def apply_deepspeed(module, optimizer, ds_config=None, log_lvl="WARNING"):
    """Apply deepspeed parallelism for given module."""
    if not ds_config:
        return None
    import deepspeed

    deepspeed.logger.setLevel(log_lvl)
    ds_model = deepspeed.initialize(None, module, optimizer, config=ds_config)[0]
    # Rollback the downcasting batchnorm stats in deepspeed initialization.
    # torch>=2.1 is required, which fixes PR#98332 for 16bit BN weight/bias.
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)
    for m in filter(lambda m: isinstance(m, bn_types), module.modules()):
        for key, buf in m._buffers.items():
            if buf is not None and "running" in key:
                num = 1.0 if "var" in key else 0.0
                m._buffers[key] = buf.float().nan_to_num(num, num, num)
    return ds_model
