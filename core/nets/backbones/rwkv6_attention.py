from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange

from fla.modules import GroupNorm, LayerNorm
from fla.models.utils import Cache
from fla.modules.activations import ACT2FN
from fla.ops.rwkv6 import chunk_rwkv6, fused_recurrent_rwkv6


if TYPE_CHECKING:
    from fla.models.utils import Cache

class RWKV6Attention(nn.Module):

    def __init__(
        self,
        mode: str = 'chunk',
        hidden_size: int = 1024,
        expand_k: float = 0.5,
        expand_v: float = 1.0,
        num_heads: int = 4,
        gate_fn: str = 'swish',
        proj_low_rank_dim: int = 32,
        gate_low_rank_dim: int = 64,
        fuse_norm: bool = True,
        elementwise_affine: Optional[bool] = True,
        norm_eps: float = 1e-5,
        layer_idx: int = None,
        **kwargs
    ) -> RWKV6Attention:
        super().__init__()

        self.mode = mode
        self.hidden_size = hidden_size
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.num_heads = num_heads
        self.proj_low_rank_dim = proj_low_rank_dim
        self.gate_low_rank_dim = gate_low_rank_dim
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.layer_idx = layer_idx

        assert mode in ['chunk', 'fused_recurrent', 'cuda'], f"Not suppoerted mode `{mode}`."
        assert self.key_dim % num_heads == 0, f"key dim must be divisible by num_heads of {num_heads}"
        assert self.value_dim % num_heads == 0, f"value dim must be divisible by num_heads of {num_heads}"

        self.head_qk_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.x_proj = nn.Sequential(
            LerpLinear(hidden_size, proj_low_rank_dim * 5),
            nn.Tanh(),
            nn.Linear(proj_low_rank_dim * 5, hidden_size, bias=False)
        )
        self.x_bias = nn.Parameter(torch.zeros(5, hidden_size))

        self.r_proj = DDLerpLinear(hidden_size, self.key_dim)
        self.w_proj = DDLerpLinear(hidden_size, self.key_dim, low_rank_dim=gate_low_rank_dim)
        self.k_proj = DDLerpLinear(hidden_size, self.key_dim)
        self.v_proj = DDLerpLinear(hidden_size, self.value_dim)
        self.g_proj = DDLerpLinear(hidden_size, self.value_dim)
        self.bonus = nn.Parameter(torch.zeros(num_heads, self.head_qk_dim))

        # TODO: fuse GroupNorm and output gate
        self.g_norm = GroupNorm(self.num_heads, self.value_dim, elementwise_affine=elementwise_affine, bias=True, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)
        self.gate_fn = ACT2FN[gate_fn]

        self.apply(self._initialize_weights)

    def _initialize_weights(self, module: nn.Module):
        if getattr(module, "_is_hf_initialized", False):
            return
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=2 ** -2.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        if isinstance(module, nn.Parameter):
            nn.init.xavier_uniform_(module, gain=2 ** -2.5)
        module._is_hf_initialized = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        batch_size, seq_len, hidden_size = hidden_states.shape
        # launching the triton kernel for just one token will actually be slower
        mode = 'fused_recurrent' if hidden_states.shape[1] == 1 else self.mode

        last_state = past_key_values[self.layer_idx] if use_cache else None

        if attention_mask is not None:
            hidden_states = hidden_states.mul_(attention_mask.unsqueeze(-1))
        if hidden_states.shape[1] == 1 and last_state is not None:
            shifted = last_state[0].unsqueeze(1)
        else:
            shifted = self.time_shift(hidden_states)
            if last_state is not None:
                shifted[:, 0] = last_state[0]

        delta = shifted - hidden_states
        x = self.x_proj[0](hidden_states, delta).view(batch_size, seq_len, -1, self.proj_low_rank_dim)
        x = torch.einsum('b l n r, h n r-> b l n h', self.x_proj[1](x), self.x_proj[2].weight.view(hidden_size, 5, -1))

        r, w, k, v, g = x.add_(self.x_bias).unbind(-2)
        r = self.r_proj(hidden_states, r, delta)
        w = self.w_proj(hidden_states, w, delta)
        k = self.k_proj(hidden_states, k, delta)
        v = self.v_proj(hidden_states, v, delta)
        g = self.g_proj(hidden_states, g, delta)

        # dealing with left-padding
        if attention_mask is not None:
            v = v.mul_(attention_mask.unsqueeze(-1))
        r, w, k, v = map(lambda x: rearrange(x, 'b l (h d) -> b h l d', h=self.num_heads), (r, w, k, v))
        w = -torch.exp(w)
        u = self.bonus

        recurrent_state = last_state[1] if use_cache else None
        if mode == 'fused_recurrent':
            o, recurrent_state = fused_recurrent_rwkv6(r, k, v, w, u,
                                                       scale=1.,
                                                       initial_state=recurrent_state,
                                                       output_final_state=use_cache)
        elif mode == 'chunk':
            o, recurrent_state = chunk_rwkv6(r, k, v, w, u,
                                             scale=1.,
                                             initial_state=recurrent_state,
                                             output_final_state=use_cache)
        elif mode == 'cuda':
            from torch.utils.cpp_extension import load
            HEAD_SIZE = self.hidden_size // self.num_heads
            load(name="wkv6", sources=["projects/mmdet3d_plugin/models/utils/cuda/wkv6_op.cpp", f"projects/mmdet3d_plugin/models/utils/cuda/wkv6_cuda.cu"], is_python_module=False,
                 verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3",
                                                  "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}",
                                                  f"-D_T_={int(30000)}"])

            class WKV_6(torch.autograd.Function):
                @staticmethod
                def forward(ctx, r, k, v, w, u):
                    with torch.no_grad():
                        B, T, C = r.size()
                        H = self.num_heads
                        assert C % H == 0
                        assert r.dtype == torch.bfloat16
                        assert k.dtype == torch.bfloat16
                        assert v.dtype == torch.bfloat16
                        assert w.dtype == torch.bfloat16
                        assert u.dtype == torch.bfloat16
                        ctx.B = B
                        ctx.T = T
                        ctx.C = C
                        ctx.H = H
                        assert r.is_contiguous()
                        assert k.is_contiguous()
                        assert v.is_contiguous()
                        assert w.is_contiguous()
                        assert u.is_contiguous()
                        ctx.save_for_backward(r, k, v, w, u)
                        y = torch.empty((B, T, C), device=r.device, dtype=torch.bfloat16,
                                        memory_format=torch.contiguous_format)  # .uniform_(-100, 100)
                        torch.ops.wkv6.forward(B, T, C, H, r, k, v, w, u, y)
                        return y

                @staticmethod
                def backward(ctx, gy):
                    with torch.no_grad():
                        assert gy.dtype == torch.bfloat16
                        B = ctx.B
                        T = ctx.T
                        C = ctx.C
                        H = ctx.H
                        assert gy.is_contiguous()
                        r, k, v, w, u = ctx.saved_tensors
                        gr = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16,
                                         memory_format=torch.contiguous_format)  # .uniform_(-100, 100)
                        gk = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16,
                                         memory_format=torch.contiguous_format)  # .uniform_(-100, 100)
                        gv = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16,
                                         memory_format=torch.contiguous_format)  # .uniform_(-100, 100)
                        gw = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16,
                                         memory_format=torch.contiguous_format)  # .uniform_(-100, 100)
                        gu = torch.empty((B, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16,
                                         memory_format=torch.contiguous_format)  # .uniform_(-100, 100)
                        torch.ops.wkv6.backward(B, T, C, H, r, k, v, w, u, gy, gr, gk, gv, gw, gu)
                        gu = torch.sum(gu, 0).view(H, C // H)
                        return (gr, gk, gv, gw, gu)

            def RUN_CUDA_RWKV6(r, k, v, w, u):
                B, num_heads, T, head_dim = r.size()
                r = r.contiguous().view(B, T,num_heads*head_dim).bfloat16()
                k = k.contiguous().view(B, T,num_heads*head_dim).bfloat16()
                v = v.contiguous().view(B, T,num_heads*head_dim).bfloat16()
                w = w.contiguous().view(B, T,num_heads*head_dim).bfloat16()
                u = u.bfloat16()

                return WKV_6.apply(r, k, v, w, u).reshape(B, num_heads, T, head_dim)
            o= RUN_CUDA_RWKV6(r, k, v, w, u)
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        if past_key_values is not None:
            past_key_values.update((hidden_states[:, -1], recurrent_state), self.layer_idx, r.shape[2])

        o = self.g_norm(rearrange(o, 'b h l d -> b l (h d)')) * self.gate_fn(g)
        o = self.o_proj(o)

        return o, None, past_key_values

    def init_state(self, batch_size: int) -> Tuple[torch.Tensor]:
        param = next(self.parameters())
        state = [param.new_zeros(batch_size, self.hidden_size),
                 param.new_zeros(batch_size, self.num_heads, self.head_qk_dim, self.head_v_dim)]
        return state

    def state_size(self, **kwargs) -> int:
        state_size = self.key_dim * self.head_v_dim
        return state_size


class RWKV6_CrossAttention(nn.Module):

    def __init__(
            self,
            mode: str = 'chunk',
            hidden_size: int = 256,
            expand_k: float = 1.0,
            expand_v: float = 1.0,
            num_heads: int = 4,
            gate_fn: str = 'swish',
            proj_low_rank_dim: int = 32,
            gate_low_rank_dim: int = 64,
            elementwise_affine: Optional[bool] = True,
            norm_eps: float = 1e-5,
            layer_idx: int = None,
            **kwargs
    ):
        super().__init__()

        self.mode = mode
        self.hidden_size = hidden_size
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.num_heads = num_heads
        self.proj_low_rank_dim = proj_low_rank_dim
        self.gate_low_rank_dim = gate_low_rank_dim

        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.layer_idx = layer_idx

        assert mode in ['chunk', 'fused_recurrent', 'cuda'], f"Not suppoerted mode `{mode}`."
        assert self.key_dim % num_heads == 0, f"key dim must be divisible by num_heads of {num_heads}"
        assert self.value_dim % num_heads == 0, f"value dim must be divisible by num_heads of {num_heads}"

        self.head_qk_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        self.rwg_proj = nn.Sequential(
            LerpLinear(hidden_size, proj_low_rank_dim * 3),
            nn.Tanh(),
            nn.Linear(proj_low_rank_dim * 3, hidden_size, bias=False)
        )
        self.rwg_bias = nn.Parameter(torch.zeros(3, hidden_size))

        self.kv_proj = nn.Sequential(
            LerpLinear(hidden_size, proj_low_rank_dim * 2),
            nn.Tanh(),
            nn.Linear(proj_low_rank_dim * 2, hidden_size, bias=False)
        )
        self.kv_bias = nn.Parameter(torch.zeros(2, hidden_size))

        self.r_proj = DDLerpLinear(hidden_size, self.key_dim)
        self.w_proj = DDLerpLinear(hidden_size, self.key_dim, low_rank_dim=gate_low_rank_dim)
        self.k_proj = DDLerpLinear(hidden_size, self.key_dim)
        self.v_proj = DDLerpLinear(hidden_size, self.value_dim)
        self.g_proj = DDLerpLinear(hidden_size, self.value_dim)
        self.bonus = nn.Parameter(torch.zeros(num_heads, self.head_qk_dim))

        self.g_norm = GroupNorm(self.num_heads, self.value_dim, elementwise_affine=elementwise_affine, bias=True,
                                eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)
        self.gate_fn = ACT2FN[gate_fn]

        self.apply(self._initialize_weights)

    def _initialize_weights(self, module: nn.Module):
        if getattr(module, "_is_hf_initialized", False):
            return
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=2 ** -2.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        if isinstance(module, nn.Parameter):
            nn.init.xavier_uniform_(module, gain=2 ** -2.5)
        module._is_hf_initialized = True

    def forward(
            self,
            query: torch.Tensor,
            keyval: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[Cache] = None,
            use_cache: Optional[bool] = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        batch_size, que_seq_len, hidden_size = query.shape
        batch_size, seq_len, hidden_size = keyval.shape

        # expand the query and the hidden_states
        expanded_query = query.unsqueeze(2).repeat(1, 1, seq_len, 1)
        expanded_query = expanded_query.reshape(-1, seq_len, hidden_size)

        hidden_states = keyval.unsqueeze(1).repeat(1, que_seq_len, 1, 1)
        hidden_states = hidden_states.reshape(-1, seq_len, hidden_size)
        expanded_batch_size = hidden_states.shape[0]

        mode = 'fused_recurrent' if hidden_states.shape[1] == 1 else self.mode
        last_state = past_key_values[self.layer_idx] if use_cache else None

        if attention_mask is not None:
            hidden_states = hidden_states.mul_(attention_mask.unsqueeze(-1))
        if hidden_states.shape[1] == 1 and last_state is not None:
            shifted = last_state[0].unsqueeze(1)
        else:
            shifted = self.time_shift(hidden_states)
            if last_state is not None:
                shifted[:, 0] = last_state[0]

        delta_cross = expanded_query - hidden_states
        delta = shifted - hidden_states

        rwg = self.rwg_proj[0](hidden_states, delta_cross).view(expanded_batch_size, seq_len, -1,
                                                                self.proj_low_rank_dim)
        rwg = torch.einsum('b l n r, h n r-> b l n h', self.rwg_proj[1](rwg),
                           self.rwg_proj[2].weight.view(hidden_size, 3, -1))
        r, w, g = rwg.add_(self.rwg_bias).unbind(-2)

        kv = self.kv_proj[0](hidden_states, delta).view(expanded_batch_size, seq_len, -1, self.proj_low_rank_dim)
        kv = torch.einsum('b l n r, h n r-> b l n h', self.kv_proj[1](kv),
                          self.kv_proj[2].weight.view(hidden_size, 2, -1))
        k, v = kv.add_(self.kv_bias).unbind(-2)

        r = self.r_proj(hidden_states, r, delta_cross)
        w = self.w_proj(hidden_states, w, delta_cross)
        g = self.g_proj(hidden_states, g, delta_cross)
        k = self.k_proj(hidden_states, k, delta)
        v = self.v_proj(hidden_states, v, delta)

        # dealing with left-padding
        if attention_mask is not None:
            v = v.mul_(attention_mask.unsqueeze(-1))
        r, w, k, v = map(lambda x: rearrange(x, 'b l (h d) -> b h l d', h=self.num_heads), (r, w, k, v))
        w = -torch.exp(w)
        u = self.bonus

        recurrent_state = last_state[1] if use_cache else None
        if mode == 'fused_recurrent':
            o, recurrent_state = fused_recurrent_rwkv6(r, k, v, w, u,
                                                       scale=1.,
                                                       initial_state=recurrent_state,
                                                       output_final_state=use_cache)
        elif mode == 'chunk':
            o, recurrent_state = chunk_rwkv6(r, k, v, w, u,
                                             scale=1.,
                                             initial_state=recurrent_state,
                                             output_final_state=use_cache)
        elif mode == 'cuda':
            from torch.utils.cpp_extension import load
            HEAD_SIZE = self.hidden_size // self.num_heads
            load(name="wkv6", sources=["projects/mmdet3d_plugin/models/utils/cuda/wkv6_op.cpp", f"projects/mmdet3d_plugin/models/utils/cuda/wkv6_cuda.cu"], is_python_module=False,
                 verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3",
                                                  "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}",
                                                  f"-D_T_={int(30000)}"])

            class WKV_6(torch.autograd.Function):
                @staticmethod
                def forward(ctx, r, k, v, w, u):
                    with torch.no_grad():
                        B, T, C = r.size()
                        H = self.num_heads
                        assert C % H == 0
                        assert r.dtype == torch.bfloat16
                        assert k.dtype == torch.bfloat16
                        assert v.dtype == torch.bfloat16
                        assert w.dtype == torch.bfloat16
                        assert u.dtype == torch.bfloat16
                        ctx.B = B
                        ctx.T = T
                        ctx.C = C
                        ctx.H = H
                        assert r.is_contiguous()
                        assert k.is_contiguous()
                        assert v.is_contiguous()
                        assert w.is_contiguous()
                        assert u.is_contiguous()
                        ctx.save_for_backward(r, k, v, w, u)
                        y = torch.empty((B, T, C), device=r.device, dtype=torch.bfloat16,
                                        memory_format=torch.contiguous_format)  # .uniform_(-100, 100)
                        torch.ops.wkv6.forward(B, T, C, H, r, k, v, w, u, y)
                        return y

                @staticmethod
                def backward(ctx, gy):
                    with torch.no_grad():
                        assert gy.dtype == torch.bfloat16
                        B = ctx.B
                        T = ctx.T
                        C = ctx.C
                        H = ctx.H
                        assert gy.is_contiguous()
                        r, k, v, w, u = ctx.saved_tensors
                        gr = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16,
                                         memory_format=torch.contiguous_format)  # .uniform_(-100, 100)
                        gk = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16,
                                         memory_format=torch.contiguous_format)  # .uniform_(-100, 100)
                        gv = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16,
                                         memory_format=torch.contiguous_format)  # .uniform_(-100, 100)
                        gw = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16,
                                         memory_format=torch.contiguous_format)  # .uniform_(-100, 100)
                        gu = torch.empty((B, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16,
                                         memory_format=torch.contiguous_format)  # .uniform_(-100, 100)
                        torch.ops.wkv6.backward(B, T, C, H, r, k, v, w, u, gy, gr, gk, gv, gw, gu)
                        gu = torch.sum(gu, 0).view(H, C // H)
                        return (gr, gk, gv, gw, gu)

            def RUN_CUDA_RWKV6(r, k, v, w, u):
                B, num_heads, T, head_dim = r.size()
                r = r.contiguous().view(B, T, num_heads * head_dim).bfloat16()
                k = k.contiguous().view(B, T, num_heads * head_dim).bfloat16()
                v = v.contiguous().view(B, T, num_heads * head_dim).bfloat16()
                w = w.contiguous().view(B, T, num_heads * head_dim).bfloat16()
                u = u.bfloat16()

                return WKV_6.apply(r, k, v, w, u).reshape(B, num_heads, T, head_dim)
            o = RUN_CUDA_RWKV6(r, k, v, w, u)
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        if past_key_values is not None:
            past_key_values.update((hidden_states[:, -1], recurrent_state), self.layer_idx, r.shape[2])

        o = self.g_norm(rearrange(o, 'b h l d -> b l (h d)')) * self.gate_fn(g)
        o = o[:, -1, :]
        o = o.view(batch_size, que_seq_len, hidden_size)
        o = self.o_proj(o)

        return o, None, past_key_values

    def init_state(self, batch_size: int) -> Tuple[torch.Tensor]:
        param = next(self.parameters())
        state = [param.new_zeros(batch_size, self.hidden_size),
                 param.new_zeros(batch_size, self.num_heads, self.head_qk_dim, self.head_v_dim)]
        return state

    def state_size(self, **kwargs) -> int:
        state_size = self.key_dim * self.head_v_dim
        return state_size


class RWKV6FeedForward(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        hidden_ratio: Optional[int] = None,
        intermediate_size: Optional[int] = None,
        hidden_act: str = 'sqrelu',
        layer_idx: int = None
    ) -> RWKV6FeedForward:
        super().__init__()

        self.hidden_size = hidden_size
        if hidden_ratio is None:
            hidden_ratio = 3.5
        if intermediate_size is None:
            intermediate_size = int(hidden_size * hidden_ratio)
            intermediate_size = 32 * ((intermediate_size + 32 - 1) // 32)
        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        self.key = LerpLinear(hidden_size, intermediate_size)
        self.value = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.receptance = LerpLinear(hidden_size, hidden_size)
        self.act_fn = ACT2FN[hidden_act]

        self.layer_idx = layer_idx

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        state: Optional[Cache] = None
    ) -> torch.Tensor:
        if attention_mask is not None:
            x = x.mul_(attention_mask.unsqueeze(-1))
        if x.shape[1] == 1 and state is not None:
            shifted = state[self.layer_idx][-1].unsqueeze(1)
        else:
            shifted = self.time_shift(x)
            if state is not None:
                shifted[:, 0] = state[self.layer_idx][-1]
        delta = shifted - x
        key = self.act_fn(self.key(x, delta))
        value = self.value(key)
        receptance = self.receptance(x, delta)

        if state is not None:
            state[self.layer_idx][-1] = x[:, -1]
        return receptance.sigmoid() * value, state

    def init_state(self, batch_size: Optional[int] = None) -> Tuple[torch.Tensor]:
        param = next(self.parameters())
        state = [param.new_zeros(batch_size, self.hidden_size)]
        return state

class LoRA(nn.Module):

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        low_rank_dim: int,
        bias: Optional[bool] = True
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.low_rank_dim = low_rank_dim
        self.bias = bias

        self.lora = nn.Sequential(
            nn.Linear(input_dim, low_rank_dim, bias=False),
            nn.Tanh(),
            nn.Linear(low_rank_dim, output_dim, bias=bias)
        )

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}("
        s += f"input_dim={self.input_dim}, low_rank_dim={self.low_rank_dim}, output_dim={self.output_dim}"
        if not self.bias:
            s += f", bias={self.bias}"
        s += ")"
        return s

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora(x)


class LerpLinear(nn.Module):

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        low_rank_dim: Optional[int] = None
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.low_rank_dim = low_rank_dim

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        if low_rank_dim is None:
            self.linear = nn.Linear(input_dim, output_dim, bias=False)
        else:
            self.linear = LoRA(input_dim, output_dim, low_rank_dim)
        self.mu = nn.Parameter(torch.zeros(input_dim))

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}({self.input_dim}, {self.output_dim}"
        if self.low_rank_dim is not None:
            s += f", low_rank_dim={self.low_rank_dim}"
        s += ")"
        return s

    def forward(self, x: torch.Tensor, delta: Optional[torch.Tensor] = None) -> torch.Tensor:
        if delta is None:
            shifted = self.time_shift(x)
            if len(shifted.shape) == 2:
                shifted = shifted.unsqueeze(1)
            delta = shifted - x
        return self.linear(x + delta * self.mu)


class DDLerpLinear(nn.Module):

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        low_rank_dim: Optional[int] = None
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.low_rank_dim = low_rank_dim

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        if low_rank_dim is None:
            self.linear = nn.Linear(input_dim, output_dim, bias=False)
        else:
            self.linear = LoRA(input_dim, output_dim, low_rank_dim)

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}({self.input_dim}, {self.output_dim}"
        if self.low_rank_dim is not None:
            s += f", low_rank_dim={self.low_rank_dim}"
        s += ")"
        return s

    def forward(self, x: torch.Tensor, mu: torch.Tensor, delta: Optional[torch.Tensor] = None) -> torch.Tensor:
        if delta is None:
            shifted = self.time_shift(x)
            if len(shifted.shape) == 2:
                shifted = shifted.unsqueeze(1)
            delta = shifted - x
        return self.linear(x + delta * mu)




