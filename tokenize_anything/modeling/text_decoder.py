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
"""Text decoder."""

try:
    from flash_attn import flash_attn_func
    from flash_attn import flash_attn_with_kvcache
    from flash_attn.layers.rotary import apply_rotary_emb
except ImportError:
    flash_attn_func = None
    flash_attn_with_kvcache = None
    apply_rotary_emb = None

import torch
from torch import nn


class TransformerCache(nn.Module):
    """Transformer cache module."""

    def __init__(self, device=None, dtype=None):
        super(TransformerCache, self).__init__()
        self.device = device
        self.dtype = dtype
        self.start_pos = 0
        self.cache_dict = {}

    def init_seq(self, max_batch_size):
        seq_lens = torch.zeros(max_batch_size, dtype=torch.int32, device=self.device)
        self.cache_dict["seq_lens"] = seq_lens

    def init_rotary(self, seq_len, dim, theta=10000.0):
        grid = torch.arange(seq_len, dtype=torch.float32).unsqueeze_(-1)
        freq = torch.pow(theta, torch.arange(0, dim, 2)[: dim // 2].float().div_(dim))
        broadcast_freq = grid.mul(freq.reciprocal_().unsqueeze_(0))
        cache_cos = broadcast_freq.cos().view((-1, dim // 2))
        cache_sin = broadcast_freq.sin().view((-1, dim // 2))
        self.cache_dict["cos"] = cache_cos.to(self.device, self.dtype)
        self.cache_dict["sin"] = cache_sin.to(self.device, self.dtype)

    def init_kv(self, mixer, kv_size):
        cache_k = torch.zeros(*kv_size, dtype=self.dtype, device=self.device)
        cache_v = torch.zeros(*kv_size, dtype=self.dtype, device=self.device)
        self.cache_dict[f"{id(mixer)}_k"] = cache_k
        self.cache_dict[f"{id(mixer)}_v"] = cache_v

    def set_seq(self, start_pos=0, end_pos=None):
        self.start_pos = start_pos
        if "seq_lens" in self.cache_dict:
            self.cache_dict["seq_lens"].fill_(start_pos)
        if "cos" in self.cache_dict and end_pos is not None:
            self.cache_dict["seq_cos"] = self.cache_dict["cos"][self.start_pos : end_pos]
            self.cache_dict["seq_sin"] = self.cache_dict["sin"][self.start_pos : end_pos]

    def forward_rotary(self, q, k, inplace=False):
        cos = self.cache_dict.get("seq_cos", self.cache_dict.get("cos", None))
        sin = self.cache_dict.get("seq_sin", self.cache_dict.get("sin", None))
        if cos is None or sin is None:
            return q, k
        q = apply_rotary_emb(q, cos, sin, interleaved=True, inplace=inplace)
        k = apply_rotary_emb(k, cos, sin, interleaved=True, inplace=inplace)
        return q, k

    def forward_flash(self, mixer, q, k, v):
        cache_k = self.cache_dict.get(f"{id(mixer)}_k", None)
        cache_v = self.cache_dict.get(f"{id(mixer)}_v", None)
        flash_args = {"softmax_scale": mixer.scale, "causal": True}
        if cache_k is None or cache_v is None:
            flash_args["dropout_p"] = mixer.dropout.p if mixer.training else 0
            return flash_attn_func(q, k, v, **flash_args)
        flash_args["cache_seqlens"] = self.cache_dict["seq_lens"][: q.shape[0]]
        return flash_attn_with_kvcache(q, cache_k, cache_v, k, v, **flash_args)


class Attention(nn.Module):
    """Self-Attention layer."""

    def __init__(self, dim, num_heads, bias=True):
        super(Attention, self).__init__()
        self.qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.proj = nn.Linear(dim, dim, bias=bias)
        self.head_dim = dim // num_heads
        self.num_heads = num_heads
        self.scale = self.head_dim**-0.5
        self.dropout = nn.Dropout(0.1, inplace=False)
        self.cache = nn.Module()

    def forward(self, x):
        qkv_shape = (-1, x.size(1), 3, self.num_heads, self.head_dim)
        q, k, v = self.qkv(x).view(qkv_shape).unbind(dim=2)
        q, k = self.cache.forward_rotary(q, k, inplace=True)
        o = self.cache.forward_flash(self, q, k, v)
        return self.proj(o.flatten(2))


class MLP(nn.Module):
    """Two layers MLP."""

    def __init__(self, dim, mlp_dim, bias=True):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(dim, mlp_dim, bias=bias)
        self.fc2 = nn.Linear(mlp_dim, dim, bias=bias)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))


class Block(nn.Module):
    """Transformer block."""

    def __init__(self, dim, num_heads, mlp_dim, bias=True):
        super(Block, self).__init__()
        self.attn = Attention(dim, num_heads, bias=bias)
        self.mlp = MLP(dim, mlp_dim, bias=bias)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(0.1, inplace=True)

    def forward(self, x):
        x = self.dropout(self.attn(self.norm1(x))).add_(x)
        return self.dropout(self.mlp(self.norm2(x))).add_(x)


class Transformer(nn.Module):
    """Causal transformer decoder."""

    def __init__(self, depth, dim, num_heads, mlp_dim, vocab_size):
        super(Transformer, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.vocab_size = vocab_size
        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList(Block(dim, num_heads, mlp_dim) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)
        self.text_proj = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, prompts, tokens, start_pos=0):
        prompt_len = prompts.size(1)
        start_pos = start_pos + (prompt_len if start_pos > 0 else 0)
        end_pos = start_pos + tokens.size(1) + (0 if start_pos > 0 else prompt_len)
        self.cache.set_seq(start_pos, end_pos)
        x = self.tok_embeddings(tokens)
        x = x if start_pos > 0 else torch.cat([prompts, x], dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x[:, 0 if start_pos > 0 else prompt_len :])
        return self.text_proj(x).float()


class TextDecoder(nn.Module):
    """Module to decode texts."""

    def __init__(
        self,
        depth,
        embed_dim,
        num_heads,
        mlp_ratio,
        prompt_embed_dim,
        max_seq_len,
        vocab_size,
    ):
        super(TextDecoder, self).__init__()
        self.max_seq_len = max_seq_len
        self.max_text_len = self.max_seq_len - 1
        self.encoder = nn.Linear(prompt_embed_dim, embed_dim, bias=False)
        self.transformer = Transformer(
            depth=depth,
            dim=embed_dim,
            mlp_dim=embed_dim * mlp_ratio,
            num_heads=num_heads,
            vocab_size=vocab_size,
        )

    def reset_cache(self, max_batch_size=1, max_seq_len=None):
        device, dtype = self.encoder.weight.device, self.encoder.weight.dtype
        max_seq_len = self.max_seq_len if max_seq_len is None else max_seq_len
        num_heads, head_dim = self.transformer.num_heads, self.transformer.head_dim
        self.transformer.cache = TransformerCache(device=device, dtype=dtype)
        self.transformer.cache.init_seq(max_batch_size)
        self.transformer.cache.init_rotary(max_seq_len, head_dim, theta=10000.0)
        kv_cache_size = (max_batch_size, max_seq_len, num_heads, head_dim)
        for blk in self.transformer.blocks:
            blk.attn.__dict__["cache"] = self.transformer.cache
            self.transformer.cache.init_kv(blk.attn, kv_cache_size) if not self.training else None

    def get_prompts(self, prompt_tokens):
        return self.encoder(prompt_tokens)

    def get_outputs(self, inputs, start_pos=0):
        return {"text_pred": self.transformer(inputs["prompts"], inputs["tokens"], start_pos)}

    def forward(self, inputs, start_pos=0):
        return self.get_outputs(inputs, start_pos)
