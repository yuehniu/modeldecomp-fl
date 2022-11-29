import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from model.resnetcifar import BasicBlock

__all__ = ['ResNet', 'resnet8' ]

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d( in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False )


'''
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d( planes, track_running_stats=False )
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d( planes, track_running_stats=False )
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
'''


class ResNet(nn.Module):

    def __init__( self, block, layers, len_feature=512, num_classes=10 ):
        n_base = 12
        self.inplanes = n_base
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d( 3, n_base, kernel_size=3, stride=1, padding=1, bias=False )
        self.bn1 = nn.BatchNorm2d( n_base, track_running_stats=False )
        self.relu = nn.ReLU( inplace=True )
        self.layer1 = self._make_layer( block, n_base,  layers[ 0 ], stride=1 )
        self.layer2 = self._make_layer( block, n_base*2, layers[ 1 ], stride=2 )
        self.layer3 = self._make_layer( block, n_base*4, layers[ 2 ], stride=2 )
        self.layer4 = self._make_layer( block, n_base*8, layers[ 3 ], stride=2 )
        self.avgpool = nn.AdaptiveAvgPool2d( 1 )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear( n_base * 8 * block.expansion, num_classes )

        for m in self.modules():
            if isinstance( m, nn.Conv2d ):
                nn.init.kaiming_normal_( m.weight, mode='fan_out', nonlinearity='relu' )
            elif isinstance( m, nn.BatchNorm2d ):
                nn.init.constant_( m.weight, 1 )
                nn.init.constant_( m.bias, 0 )

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d( self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False ),
                nn.BatchNorm2d( planes * block.expansion, track_running_stats=False ),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward( self, x ):
        x = self.conv1( x )
        x = self.bn1( x )
        x = self.relu( x )

        x = self.layer1( x )
        x = self.layer2( x )
        x = self.layer3( x )
        x = self.layer4( x )

        x = self.avgpool( x )
        # x = x.view(x.size(0), -1)
        x = self.flatten( x )
        x = self.fc( x )

        return x


def init_dist_weights(model):
    # https://arxiv.org/pdf/1706.02677.pdf
    # https://github.com/pytorch/examples/pull/262
    for m in model.modules():
        if isinstance( m, BasicBlock ): m.bn2.weight = nn.Parameter( torch.zeros_like( m.bn2.weight ) )
        if isinstance( m, nn.Linear ): m.weight.data.normal_( 0, 0.01 )


def resnet8(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet( BasicBlock, [2, 2, 2, 2], **kwargs )
    if pretrained:  # no pretrained model for resnet8
        pass
    return model
