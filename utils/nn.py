"""Utility function in define neural networks

Note:
"""
from model.vgg import vgg11, vgg11_bn, vgg16, vgg16_bn, vgg19, vgg19_bn
from model.resnetcifar import resnet18, resnet34

def build_network( model_name, **kwargs ):
    models = {
        'vgg11': vgg11,
        'vgg11_bn': vgg11_bn,
        'vgg16': vgg16,
        'vgg16_bn': vgg16_bn,
        'vgg19': vgg19,
        'vgg19_bn': vgg19_bn,
        'resnet18': resnet18,
        'resnet34': resnet34
    }

    return models[ model_name ]( **kwargs )



def convert_to_principal_model:
    pass