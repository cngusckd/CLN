# # Code from mamamoth


# from typing import List

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.nn.functional import avg_pool2d, relu


# # 3x3 Convolution
# def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> F.conv2d:
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
#                      padding=1, bias=False)

# # Residual Block
# class BasicBlock(nn.Module):

#     expansion = 1

#     def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:

#         super(BasicBlock, self).__init__()
#         self.return_prerelu = False
#         self.conv1 = conv3x3(in_planes, planes, stride)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = nn.BatchNorm2d(planes)

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion * planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
#                           stride=stride, bias=False),
#                 nn.BatchNorm2d(self.expansion * planes)
#             )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         out = relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += self.shortcut(x)

#         if self.return_prerelu:
#             self.prerelu = out.clone()

#         out = relu(out)
#         return out

# # Bottleneck Block
# class Bottleneck(nn.Module):
#     expansion = 4

#     def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
#         super(Bottleneck, self).__init__()
#         self.return_prerelu = False
#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(self.expansion * planes)

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion * planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(self.expansion * planes)
#             )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         out = relu(self.bn1(self.conv1(x)))
#         out = relu(self.bn2(self.conv2(out)))
#         out = self.bn3(self.conv3(out))
#         out += self.shortcut(x)

#         if self.return_prerelu:
#             self.prerelu = out.clone()

#         out = relu(out)
#         return out

# class ResNet(nn.Module):


#     def __init__(self, block: BasicBlock, num_blocks: List[int],
#                  num_classes: int, nf: int) -> None:

#         super(ResNet, self).__init__()
#         self.return_prerelu = False
#         self.device = "cpu"
#         self.in_planes = nf
#         self.block = block
#         self.num_classes = num_classes
#         self.nf = nf
#         self.conv1 = conv3x3(3, nf * 1)
#         self.bn1 = nn.BatchNorm2d(nf * 1)
#         self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
#         self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
#         self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
#         self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
#         self.classifier = nn.Linear(nf * 8 * block.expansion, num_classes)

#     def to(self, device, **kwargs):
#         self.device = device
#         return super().to(device, **kwargs)

#     def set_return_prerelu(self, enable=True):
#         self.return_prerelu = enable
#         for c in self.modules():
#             if isinstance(c, self.block):
#                 c.return_prerelu = enable

#     def _make_layer(self, block: BasicBlock, planes: int,
#                     num_blocks: int, stride: int) -> nn.Module:

#         strides = [stride] + [1] * (num_blocks - 1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_planes, planes, stride))
#             self.in_planes = planes * block.expansion
#         return nn.Sequential(*layers)

#     def forward(self, x: torch.Tensor, returnt='out') -> torch.Tensor:

#         out_0 = self.bn1(self.conv1(x))  # 64, 32, 32
#         if self.return_prerelu:
#             out_0_t = out_0.clone()
#         out_0 = relu(out_0)
#         if hasattr(self, 'maxpool'):
#             out_0 = self.maxpool(out_0)

#         out_1 = self.layer1(out_0)  # -> 64, 32, 32
#         out_2 = self.layer2(out_1)  # -> 128, 16, 16
#         out_3 = self.layer3(out_2)  # -> 256, 8, 8
#         out_4 = self.layer4(out_3)  # -> 512, 4, 4

#         feature = avg_pool2d(out_4, out_4.shape[2])  # -> 512, 1, 1
#         feature = feature.view(feature.size(0), -1)  # 512

#         if returnt == 'features':
#             return feature

#         out = self.classifier(feature)

#         if returnt == 'out':
#             return out
#         elif returnt == 'both':
#             return (out, feature)
#         elif returnt == 'full':
#             return out, [
#                 out_0 if not self.return_prerelu else out_0_t,
#                 out_1 if not self.return_prerelu else self.layer1[-1].prerelu,
#                 out_2 if not self.return_prerelu else self.layer2[-1].prerelu,
#                 out_3 if not self.return_prerelu else self.layer3[-1].prerelu,
#                 out_4 if not self.return_prerelu else self.layer4[-1].prerelu
#             ]

#         raise NotImplementedError("Unknown return type. Must be in ['out', 'features', 'both', 'all'] but got {}".format(returnt))


# def resnet18(nclasses: int, nf: int = 64) -> ResNet:
#     return ResNet(BasicBlock, [2, 2, 2, 2], nclasses, nf)

# def resnet34(nclasses: int, nf: int = 64) -> ResNet:
#     return ResNet(BasicBlock, [3, 4, 6, 3], nclasses, nf)

# def resnet50(nclasses: int, nf: int = 64) -> ResNet:
#     return ResNet(Bottleneck, [3, 4, 6, 3], nclasses, nf)

import torch
import torch.nn as nn
import torch.nn.functional as F

import os


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.bn1(out))

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def size(self):
        torch.save(self.state_dict(), "temp.p")
        size = os.path.getsize("temp.p") / 1e6
        os.remove("temp.p")
        return size


def resnet18(num_classes) -> ResNet:
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes = num_classes)

# def load_model(path: str = "{}/model.0047.pth".format(WEIGHTS_DIR)) -> ResNet:
def load_model(path: str) -> ResNet:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet18()
    
    # 저장된 체크포인트에서 모델 가중치만 로드
    checkpoint = torch.load(path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)  # 단순 state_dict인 경우
    
    return model


__all__ = ["resnet18", "load_model"]