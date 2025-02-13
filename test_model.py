import torch

from models.DSFD_ours import DSFD, build_net_resnet


x = torch.rand(1, 3, 640, 640)

model = build_net_resnet('train', 2, 'resnet50')

y = model(x)