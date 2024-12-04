import torch
from core.nets.heads import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = HeatmapHead(256, 3, 11, True, 32).to(device)

feature = torch.randn(1, 256, 40, 40).to(device)
heatmap = model(feature)
print(heatmap.shape)