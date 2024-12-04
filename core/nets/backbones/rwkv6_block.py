from __future__ import annotations

import math
import warnings
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from fla.modules import GroupNorm, LayerNorm
from fla.models.utils import Cache
from .drop import DropPath

from .rwkv6_attention import RWKV6Attention, RWKV6_CrossAttention, RWKV6FeedForward

class RWKV6Block(nn.Module):
    def __init__(self,
                 hidden_size: int = 2048,
                 norm_first: bool = True,
                 norm_bias: bool = True,
                 norm_eps: float = 1e-5,
                 attn_mode: str = "chunk",
                 expand_k: int = 1,
                 expand_v: int = 1,
                 num_heads: int = 4,
                 proj_low_rank_dim: int = 32,
                 gate_low_rank_dim: int = 64,
                 fuse_norm: bool = True,
                 hidden_ratio: Optional[int] = 3.5,
                 intermediate_size: Optional[int] = None,
                 hidden_act: str = "sqrelu",
                 use_cache: bool = False,
                 layer_idx: int = 0,
                 drop_path: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.layer_idx = layer_idx
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if norm_first and layer_idx == 0:
            self.pre_norm = LayerNorm(hidden_size = hidden_size, bias=norm_bias, eps=norm_eps)
        self.attn_norm = LayerNorm(hidden_size=hidden_size, bias=norm_bias, eps=norm_eps)
        self.attn = RWKV6Attention(
            mode=attn_mode,
            hidden_size=hidden_size,
            expand_k=expand_k,
            expand_v=expand_v,
            num_heads=num_heads,
            proj_low_rank_dim=proj_low_rank_dim,
            gate_low_rank_dim=gate_low_rank_dim,
            norm_eps=norm_eps,
            fuse_norm=fuse_norm,
            layer_idx=layer_idx
        )
        self.ffn_norm = LayerNorm(hidden_size, norm_bias, norm_eps)
        self.ffn = RWKV6FeedForward(
            hidden_size=hidden_size,
            hidden_ratio=hidden_ratio,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            layer_idx=layer_idx
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        residual = self.pre_norm(hidden_states) if hasattr(self, 'pre_norm') else hidden_states
        hidden_states = self.attn_norm(residual)
        hidden_states, attentions, past_key_values = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions
        )
        hidden_states, residual = self.ffn_norm(hidden_states, residual, True)
        hidden_states, past_key_values = self.ffn(hidden_states, attention_mask, past_key_values)
        hidden_states = residual + hidden_states

        outputs = (hidden_states, attentions, past_key_values)

        return outputs

    def init_state(self, **kwargs) -> Tuple[torch.Tensor]:
        state = []
        if callable(getattr(self.attn, 'init_state', None)):
            state += self.attn.init_state(**kwargs)
        if callable(getattr(self.ffn, 'init_state', None)):
            state += self.ffn.init_state(**kwargs)
        return state



class RWKV6_CrossAttBlock(nn.Module):
    def __init__(self,
                 hidden_size: int = 2048,
                 norm_first: bool = True,
                 norm_bias: bool = True,
                 norm_eps: float = 1e-5,
                 attn_mode: str = "chunk",
                 expand_k: int = 1,
                 expand_v: int = 1,
                 num_heads: int = 4,
                 proj_low_rank_dim: int = 32,
                 gate_low_rank_dim: int = 64,
                 fuse_norm: bool = True,
                 hidden_ratio: Optional[int] = 3.5,
                 intermediate_size: Optional[int] = None,
                 hidden_act: str = "sqrelu",
                 use_cache: bool = False,
                 layer_idx: int = 0,
                 drop_path: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.layer_idx = layer_idx
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if norm_first and layer_idx == 0:
            self.pre_norm = LayerNorm(hidden_size=hidden_size, bias=norm_bias, eps=norm_eps)
        self.query_norm = LayerNorm(hidden_size=hidden_size, bias=norm_bias, eps=norm_eps)
        self.keyval_norm = LayerNorm(hidden_size=hidden_size, bias=norm_bias, eps=norm_eps)

        self.self_attn = RWKV6Attention(
            mode=attn_mode,
            hidden_size=hidden_size,
            expand_k=expand_k,
            expand_v=expand_v,
            num_heads=num_heads,
            proj_low_rank_dim=proj_low_rank_dim,
            gate_low_rank_dim=gate_low_rank_dim,
            norm_eps=norm_eps,
            layer_idx=layer_idx
        )
        self.cross_attn = RWKV6_CrossAttention(
            mode=attn_mode,
            hidden_size=hidden_size,
            expand_k=expand_k,
            expand_v=expand_v,
            num_heads=num_heads,
            proj_low_rank_dim=proj_low_rank_dim,
            gate_low_rank_dim=gate_low_rank_dim,
            norm_eps=norm_eps,
            layer_idx=layer_idx
        )
        self.ffn_norm = LayerNorm(hidden_size=hidden_size, bias=norm_bias, eps=norm_eps)
        self.ffn = RWKV6FeedForward(
            hidden_size=hidden_size,
            hidden_ratio=hidden_ratio,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            layer_idx=layer_idx
        )

    def forward(
        self,
        query: torch.Tensor,
        keyval: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_query: Optional[Cache] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = self.pre_norm(query) if hasattr(self, 'pre_norm') else query
        query = self.query_norm(residual)
        keyval = self.keyval_norm(keyval)

        query, attentions, past_query = self.self_attn(
            hidden_states=query,
            attention_mask=attention_mask,
            past_key_values=past_query,
            use_cache=use_cache
        )
        query, attentions, past_key_values = self.cross_attn(
            query=query,
            keyval=keyval,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache
        )
        query, residual = self.ffn_norm(query, residual, True)
        query, past_key_values = self.ffn(query, attention_mask, past_key_values)
        query = residual + query

        outputs = (query, attentions, past_query, past_key_values)

        return outputs

    def init_state(self, **kwargs) -> Tuple[torch.Tensor]:
        state = []
        if callable(getattr(self.attn, 'init_state', None)):
            state += self.attn.init_state(**kwargs)
        if callable(getattr(self.ffn, 'init_state', None)):
            state += self.ffn.init_state(**kwargs)
        return state


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from types import SimpleNamespace

    # 将字典转换为 SimpleNamespace 对象
    encoder = SimpleNamespace(
        num_layers=6,
        hidden_size=256,
        norm_first=True,
        norm_bias=True,
        norm_eps=1e-5,
        attn_mode="chunk",
        expand_k=1,
        expand_v=1,
        num_heads=8,
        proj_low_rank_dim=32,
        gate_low_rank_dim=64,
        fuse_norm=True,
        hidden_ratio=3.5,
        hidden_act="sqrelu",
        use_cache=False,
        drop_path_rate=0.5),
    decoder = SimpleNamespace(
        num_layers=6,
        hidden_size=256,
        norm_first=True,
        norm_bias=True,
        norm_eps=1e-5,
        attn_mode="chunk",
        expand_k=1,
        expand_v=1,
        num_heads=8,
        proj_low_rank_dim=32,
        gate_low_rank_dim=64,
        fuse_norm=True,
        hidden_ratio=3.5,
        hidden_act="sqrelu",
        use_cache=False,
        drop_path_rate=0.5,
        return_intermediate=True),
    model = nn.ModuleList(
        [
            RWKV6Block(encoder, layer_idx)
            for layer_idx in range(encoder.num_layers)
        ])
    model.to(device)

    B = 64
    que_seq_len = 31
    seq_len = 65

    query = torch.randn(B, que_seq_len, 256).to(device)
    memory = torch.randn(B, seq_len, 256).to(device)

    for layer in model:
        memory, _, past_key_values = layer(memory, use_cache=False)

    print("Output Shape: ", memory.shape)
    print("Past Key Values: ", past_key_values)