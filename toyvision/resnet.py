import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class MiniResNet(nn.Module):
    """Mini ResNet for small images
    """
    def __init__(self,
                 inplanes: int,
                 planes: int,
                 expansion: int,
                 layers: int,
                 outplanes: int,
                 reduce_spatial: bool = True):
        super(MiniResNet, self).__init__()
        self.layers = layers
        self.inplanes = inplanes
        self.planes = planes
        self.expansion = expansion
        self.outplanes = outplanes
        self.reduce_spatial = reduce_spatial

        self.layer_0 = nn.Sequential(nn.Conv2d(inplanes, planes, 1, 1, 0),
                                     nn.ReLU(), nn.BatchNorm2d(planes))

        for layer in range(layers - 1):
            setattr(
                self, f"layer_{layer + 1}",
                nn.Sequential(
                    torchvision.models.resnet.BasicBlock(planes, planes),
                    nn.MaxPool2d(2, 2, 0),
                    nn.Conv2d(planes, planes * expansion, 1, 1, 0)))
            planes *= expansion

        if reduce_spatial:
            self.embedding = nn.Linear(planes, outplanes)
        else:
            self.embedding = nn.Conv2d(planes, outplanes, 1, 1, 0)

    def forward(self, x):

        for layer in range(self.layers):
            layer_name = f"layer_{layer}"
            x = getattr(self, layer_name)(x)

        if self.reduce_spatial:
            x = F.adaptive_avg_pool2d(x, 1)[..., 0, 0]
            x = self.embedding(x)
        else:
            x = self.embedding(x)

        return x

    def number_of_parameters(self):
        n = 0
        for p in self.parameters():
            n += p.numel()
        return n

    def _embedding_norm_regularization_loss(self,
                                            x,
                                            sigma: float = 1.0,
                                            eps: float = 1e-8):
        n_dim = self.outplanes
        r_star = sigma * math.sqrt(n_dim - 1)
        r = x.norm(p=2.0, dim=-1)
        z2 = (r / r_star + 1e-8).log()**2 * (2 * (n_dim - 1))

        return z2
