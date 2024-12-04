import torch
import torch.nn as nn




# 使用示例
if __name__ == "__main__":
    # 任意通道数的输入
    in_channels = 256
    model = MultiScaleFeatureExtractor(in_channels)
    x = torch.randn(4, in_channels, 64, 96)
    features = model(x)

    for i, feat in enumerate(features):
        print(f"Feature map {i + 1} shape: {feat.shape}")