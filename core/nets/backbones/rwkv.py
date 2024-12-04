import math
import copy
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, List, Sequence
from einops import rearrange

from .rwkv6_block import RWKV6Block, RWKV6_CrossAttBlock
from fla.models.utils import Cache
from functools import partial
import copy
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from .PatchEmbed import TimmPatchEmbed

class RWKVEncoder(nn.Module):
    def __init__(self,
                 num_layers: int = 6,
                 attn_mode: str = 'chunk',
                 hidden_size: int = 1024,
                 expand_k: float = 0.5,
                 expand_v: float = 1.0,
                 num_heads: int = 4,
                 gate_fn: str = 'swish',
                 proj_low_rank_dim: int = 32,
                 gate_low_rank_dim: int = 64,
                 fuse_norm: bool = True,
                 elementwise_affine: Optional[bool] = True,
                 norm_bias: bool = True,
                 norm_eps: float = 1e-5,
                 hidden_act: str = "sqrelu",
                 use_cache: bool = False,
                 hidden_ratio: Optional[int] = None,
                 norm_first: bool = False,
                 drop_path_rate: float = 0.5):
        super().__init__()
        self.num_layers = num_layers
        self.drop_path_rate = drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        self.layers = nn.ModuleList(
            [copy.deepcopy(
                RWKV6Block(
                    hidden_size=hidden_size,
                    norm_first=norm_first,
                    norm_bias=norm_bias,
                    norm_eps=norm_eps,
                    attn_mode=attn_mode,
                    expand_k=expand_k,
                    expand_v=expand_v,
                    num_heads=num_heads,
                    proj_low_rank_dim=proj_low_rank_dim,
                    gate_low_rank_dim=gate_low_rank_dim,
                    fuse_norm=fuse_norm,
                    hidden_ratio=hidden_ratio,
                    hidden_act=hidden_act,
                    use_cache=use_cache,
                    layer_idx=i,
                    drop_path=dpr[i]))
                for i in range(self.num_layers)])
        self.final_norm = nn.LayerNorm(hidden_size)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, x,
                pos_embed: Optional[Tensor] = None,
                past_key_values=None,
                use_cache=False):
        def checkpoint_layer(layer, x, past_key_values, use_cache):
            return layer(x, past_key_values=past_key_values, use_cache=use_cache)

        for i in range(self.num_layers):
            layer = self.layers[i]
            x = self.with_pos_embed(x, pos_embed)

            do_reverse = (i % 2 == 1)
            if do_reverse:
                x = x.flip(1)

            checkpoint_fn = partial(checkpoint_layer, layer, past_key_values=past_key_values, use_cache=use_cache)
            x, _, past_key_values = torch_checkpoint(checkpoint_fn, x, use_reentrant=False)
            '''
            x, _ , past_key_values= layer(x, past_key_values=past_key_values, use_cache=use_cache)
            '''
            if do_reverse:
                x = x.flip(1)
            if torch.isnan(x).any():
                raise ValueError("Tensor contains NaN values")
        x = self.final_norm(x)

        return x


class RWKVDecoder(nn.Module):
    def __init__(self,
                 intermediate_size: Optional[int] = None,
                 hidden_act: str = "sqrelu",
                 use_cache: bool = False,
                 num_layers: int = 6,
                 attn_mode: str = 'chunk',
                 hidden_size: int = 1024,
                 expand_k: float = 0.5,
                 expand_v: float = 1.0,
                 num_heads: int = 4,
                 gate_fn: str = 'swish',
                 proj_low_rank_dim: int = 32,
                 gate_low_rank_dim: int = 64,
                 fuse_norm: bool = True,
                 elementwise_affine: Optional[bool] = True,
                 norm_bias: bool = True,
                 norm_eps: float = 1e-5,
                 hidden_ratio: Optional[int] = None,
                 norm_first: bool = False,
                 drop_path_rate: float = 0.5,
                 return_intermediate=False):
        super().__init__()
        self.num_layers = num_layers
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        self.layers = nn.ModuleList(
            [copy.deepcopy(
                RWKV6_CrossAttBlock(
                    hidden_size=hidden_size,
                    norm_first=norm_first,
                    norm_bias=norm_bias,
                    norm_eps=norm_eps,
                    attn_mode=attn_mode,
                    expand_k=expand_k,
                    expand_v=expand_v,
                    num_heads=num_heads,
                    proj_low_rank_dim=proj_low_rank_dim,
                    gate_low_rank_dim=gate_low_rank_dim,
                    fuse_norm=fuse_norm,
                    hidden_ratio=hidden_ratio,
                    intermediate_size=intermediate_size,
                    hidden_act=hidden_act,
                    use_cache=use_cache,
                    layer_idx=i,
                    drop_path=dpr[i]))
                for i in range(self.num_layers)])
        self.norm = nn.LayerNorm(hidden_size)
        self.return_intermediate = return_intermediate

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(
            self, memory: Tensor, query: Tensor, query_pos: Optional[Tensor] = None,
            past_query: Optional[Cache] = None,
            past_key_values: Optional[Cache] = None,
            use_cache: Optional[bool] = False):
        def checkpoint_layer(layer, x, memory, past_query, past_key_values, use_cache):
            return layer(x, keyval=memory, past_query=past_query, past_key_values=past_key_values, use_cache=use_cache)

        intermediate = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            x = self.with_pos_embed(query, query_pos)

            do_reverse = (i % 2 == 1)
            if do_reverse:
                memory = memory.flip(1)

            checkpoint_fn = partial(checkpoint_layer, layer, past_query=past_query, past_key_values=past_key_values,
                                    use_cache=use_cache)
            x, _, past_query, past_key_values = torch_checkpoint(checkpoint_fn, x, memory, use_reentrant=False)
            '''
            x, _, past_query, past_key_values = layer(x, memory, past_query = past_query, past_key_values=past_key_values, use_cache=use_cache)
            '''
            if do_reverse:
                memory = memory.flip(1)
            if torch.isnan(x).any():
                raise ValueError("Tensor contains NaN values")

            if self.return_intermediate:
                intermediate.append(self.norm(x))
        if self.return_intermediate:
            return torch.stack(intermediate)
        return x.unsqueeze(0)


class RWKV(nn.Module):
    def __init__(self,
                 encoder=None,
                 feature_layers=[0,1,2],
                 feat_size = [96,64],
                 decoder=None,
                 init_cfg=None):
        super().__init__()
        if encoder is not None:
            self.encoder = RWKVEncoder(**encoder)
            self.num_encoder_layers = encoder.get('num_layers', 0)  # 从字典提取num_layers
        else:
            self.encoder = None

        self.d_model = encoder.get('hidden_size', 0)  # 从字典提取hidden_size
        self.patchembed =  TimmPatchEmbed(
            model_name='resnet50',
            pretrained=True,
            embed_dim=self.d_model,
            feature_layers=feature_layers,  # 使用最后两层
            fusion_mode='concat',
            use_norm=True,
            freeze_backbone=True,
            target_size= feat_size # 指定目标尺寸
        )
        self.nhead = encoder.get('num_heads', 0)
        self.feature_size = feat_size
        self.init_weights()

    def init_weights(self):
        # follow the official DETR to init parameters
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):

        patch_embed = self.patchembed(x)
        feat_H, feat_W = self.feature_size
        x_rwkv = self.encoder(patch_embed)
        B, C = x_rwkv.shape[0], x_rwkv.shape[-1]
        x_rwkv = x_rwkv.view(B, C, feat_W, feat_H)
        return x_rwkv




if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from types import SimpleNamespace

    x = torch.randn(1, 3, 1920, 1200).to(device) # 输入图像

    # 将字典转换为 SimpleNamespace
    encoder = {
        "num_layers": 6,
        "hidden_size": 256,
        "norm_first": True,
        "norm_bias": True,
        "norm_eps": 1e-5,
        "attn_mode": "chunk",
        "expand_k": 1,
        "expand_v": 1,
        "num_heads": 8,
        "proj_low_rank_dim": 32,
        "gate_low_rank_dim": 64,
        "fuse_norm": True,
        "hidden_ratio": 3.5,
        "hidden_act": "sqrelu",
        "use_cache": False,
        "drop_path_rate": 0.5,
    }

    decoder = {
        "num_layers": 6,
        "hidden_size": 256,
        "norm_first": True,
        "norm_bias": True,
        "norm_eps": 1e-5,
        "attn_mode": "chunk",
        "expand_k": 1,
        "expand_v": 1,
        "num_heads": 8,
        "proj_low_rank_dim": 32,
        "gate_low_rank_dim": 64,
        "fuse_norm": True,
        "hidden_ratio": 3.5,
        "hidden_act": "sqrelu",
        "use_cache": False,
        "drop_path_rate": 0.5,
        "return_intermediate": True,
    }

    model = RWKV(encoder).to(device)
    out = model(x)
    print(out.shape)  # torch.Size([1, 256, 60, 38])
