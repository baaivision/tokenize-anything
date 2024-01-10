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
import numpy as np

from tokenize_anything.utils import logging

GLOBAL_DDP_GROUP = None


def count_params(module, trainable=True, unit="M"):
    """Return the number of parameters."""
    counts = [v.size().numel() for v in module.parameters() if v.requires_grad or (not trainable)]
    return sum(counts) / {"M": 1e6, "B": 1e9}[unit]


def freeze_module(module):
    """Freeze parameters of given module."""
    module.eval()
    for param in module.parameters():
        param.requires_grad = False


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


def get_param_groups(module, layer_lr_decay=1.0):
    """Separate parameters into groups."""
    memo, groups = {}, collections.OrderedDict()
    lr_scale_getter = None
    if layer_lr_decay < 1.0 and hasattr(module.image_encoder, "get_lr_scale"):
        lr_scale_getter = functools.partial(module.image_encoder.get_lr_scale, decay=layer_lr_decay)
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        attrs = collections.OrderedDict()
        if lr_scale_getter:
            attrs["lr_scale"] = lr_scale_getter(name)
        memo[name] = param.shape
        no_weight_decay = not (name.endswith("weight") and param.dim() > 1)
        no_weight_decay = getattr(param, "no_weight_decay", no_weight_decay)
        if no_weight_decay:
            attrs["weight_decay"] = 0
        group_name = "/".join(["%s:%s" % (v[0], v[1]) for v in list(attrs.items())])
        if group_name not in groups:
            groups[group_name] = {"params": []}
            groups[group_name].update(attrs)
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


def create_ddp_group(cfg, ranks=None, devices=None, num_nodes=1):
    """Create group for data parallelism."""
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")
    world_rank = torch.distributed.get_rank()
    ranks = ranks if ranks else [i for i in range(cfg.NUM_GPUS)]
    logging.set_root(world_rank == ranks[0])
    devices_per_node = len(ranks) // num_nodes
    devices = devices if devices else [i % devices_per_node for i in range(len(ranks))]
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
