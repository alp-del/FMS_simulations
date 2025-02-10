import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

__all__ = [
    'ResNet', 'resnet8', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 
    'resnet110', 'resnet1202', 
    'ResNetTinyImageNet', 'resnet18_tinyimagenet'
]

def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

######################################################################
# 1) Original CIFAR-Style BasicBlock and ResNet
######################################################################
class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
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
        if stride != 1 or in_planes != planes:
            if option == 'A':
                # For CIFAR10 ResNet, paper uses option A
                self.shortcut = LambdaLayer(lambda x:
                    F.pad(x[:, :, ::2, ::2],
                          (0, 0, 0, 0, planes // 4, planes // 4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        in_planes, self.expansion * planes,
                        kernel_size=1, stride=stride, bias=False
                    ),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """
    CIFAR-like ResNet with only 3 stages (layer1, layer2, layer3).
    """
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(
            3, 16, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def resnet8(num_classes=10):
    return ResNet(BasicBlock, [1, 1, 1], num_classes)

def resnet20(num_classes=10):
    return ResNet(BasicBlock, [3, 3, 3], num_classes)

def resnet32(num_classes=10):
    return ResNet(BasicBlock, [5, 5, 5], num_classes)

def resnet44(num_classes=10):
    return ResNet(BasicBlock, [7, 7, 7], num_classes)

def resnet56(num_classes=10):
    return ResNet(BasicBlock, [9, 9, 9], num_classes)

def resnet110(num_classes=10):
    return ResNet(BasicBlock, [18, 18, 18], num_classes)

def resnet1202(num_classes=10):
    return ResNet(BasicBlock, [200, 200, 200], num_classes)

######################################################################
# 2) NEW: ImageNet-Style BasicBlock + Tiny-ImageNet ResNet-18
######################################################################
class BasicBlockIM(nn.Module):
    """
    Standard ResNet BasicBlock for ImageNet-like usage (ResNet-18/34),
    using explicit downsample for stride != 1 or channel mismatch.
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlockIM, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out, inplace=True)
        return out


class ResNetTinyImageNet(nn.Module):
    """
    ResNet architecture for Tiny ImageNet (64x64).
    This implements a ResNet-18 style network:
      - initial 7x7 conv (stride=2) + maxpool(3x3, stride=2)
      - 4 layers, each with BasicBlockIM
      - final linear for num_classes
    """
    def __init__(self, block, layers, num_classes=200):
        super(ResNetTinyImageNet, self).__init__()
        self.in_planes = 64

        # ImageNet-style 'stem':
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 4 layers: [2, 2, 2, 2] for ResNet-18
        self.layer1 = self._make_layer(block, 64,  layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._initialize_weights()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        # If we need to downsample (stride=2 or channel mismatch), build 1x1 conv
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_planes, planes * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def resnet18_tinyimagenet(num_classes=200):
    """
    Builds a ResNet-18 for Tiny ImageNet (64x64 images).
    """
    return ResNetTinyImageNet(
        block=BasicBlockIM, 
        layers=[2, 2, 2, 2],
        num_classes=num_classes
    )

######################################################################
# 3) Quick test function
######################################################################
def test(net):
    import numpy as np
    total_params = 0
    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.cpu().numpy().shape)
    print("Total number of params: ", total_params)
    print("Total layers: ", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))

if __name__ == "__main__":
    # Test the original CIFAR ResNets
    for net_name in __all__:
        if net_name.startswith('resnet') and net_name not in ('resnet18_tinyimagenet',):
            print(net_name)
            net = globals()[net_name]()
            test(net)
            print()

    # Test the new Tiny-ImageNet ResNet-18
    print("resnet18_tinyimagenet")
    tiny_net = resnet18_tinyimagenet(num_classes=200)
    test(tiny_net)
