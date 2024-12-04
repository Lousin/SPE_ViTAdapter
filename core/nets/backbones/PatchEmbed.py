import torch
import torch.nn as nn
import timm
from typing import List, Union, Optional


class TimmPatchEmbed(nn.Module):
    def __init__(
            self,
            model_name: str = "resnet50",
            pretrained: bool = True,
            embed_dim: int = 768,
            feature_layers: Union[int, List[int]] = -1,
            fusion_mode: str = 'add',
            use_norm: bool = True,
            freeze_backbone: bool = True,
            target_size: Optional[tuple] = None  # 添加目标尺寸参数
    ):
        super().__init__()

        assert fusion_mode in ['add', 'concat'], "fusion_mode must be 'add' or 'concat'"
        self.fusion_mode = fusion_mode
        self.use_norm = use_norm
        self.target_size = target_size


        if isinstance(feature_layers, int):
            feature_layers = [feature_layers]
        self.feature_layers = feature_layers


        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True
        )


        self.out_channels = [
            self.backbone.feature_info[i]['num_chs']
            for i in range(len(self.backbone.feature_info))
        ]


        self.projectors = nn.ModuleDict()
        for idx in feature_layers:
            if idx < 0:  # 处理负索引
                idx = len(self.out_channels) + idx
            self.projectors[f'proj_{idx}'] = nn.Sequential(
                nn.Conv2d(self.out_channels[idx], embed_dim, kernel_size=1),
                nn.BatchNorm2d(embed_dim) if use_norm else nn.Identity()
            )

        if fusion_mode == 'concat' and len(feature_layers) > 1:
            self.final_proj = nn.Linear(embed_dim * len(feature_layers), embed_dim)
        else:
            self.final_proj = None

        self.embed_dim = embed_dim

        # 如果需要，冻结backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def _interpolate_features(self, features_list: List[torch.Tensor], target_size: tuple) -> List[torch.Tensor]:
        """将所有特征图插值到指定的大小"""
        return [
            nn.functional.interpolate(
                feat,
                size=target_size,
                mode='bilinear',
                align_corners=False
            ) if feat.shape[-2:] != target_size else feat
            for feat in features_list
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 获取输入大小
        B, C, H, W = x.shape

        # 获取所有层的特征
        features = self.backbone(x)

        projected_features = []
        for i, layer_idx in enumerate(self.feature_layers):
            if layer_idx < 0:
                layer_idx = len(features) + layer_idx
            feat = features[layer_idx]
            proj_feat = self.projectors[f'proj_{layer_idx}'](feat)
            projected_features.append(proj_feat)

        if self.target_size is not None:
            target_size = self.target_size
        else:
            min_h = min(feat.shape[2] for feat in projected_features)
            min_w = min(feat.shape[3] for feat in projected_features)
            target_size = (min_h, min_w)

        # 对齐特征尺寸
        aligned_features = self._interpolate_features(projected_features, target_size)

        # 检查对齐后的特征尺寸
        for i, feat in enumerate(aligned_features):
            assert feat.shape[2:] == aligned_features[0].shape[2:], \
                f"Feature {i} shape {feat.shape} doesn't match target shape {aligned_features[0].shape}"

        # 融合特征
        if self.fusion_mode == 'add' or len(aligned_features) == 1:
            x = sum(aligned_features)
        else:  # concat模式
            x = torch.cat(aligned_features, dim=1)
            B, C, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
            x = self.final_proj(x)  # (B, H*W, embed_dim)
            x = x.transpose(1, 2).reshape(B, self.embed_dim, H, W)

        x = x.flatten(2).transpose(1, 2)

        return x


if __name__ == "__main__":
    # 多层示例
    model_multi = TimmPatchEmbed(
        model_name='resnet50',
        pretrained=True,
        embed_dim=256,
        feature_layers=[0, 1, 2],
        fusion_mode='concat',
        use_norm=True,
        freeze_backbone=True,
        target_size=None  # 指定目标尺寸
    )

    # 测试
    x = torch.randn(1, 3, 768, 512)
    out_multi = model_multi(x)
    print(f"Multi layer output shape: {out_multi.shape}")