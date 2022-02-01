"""Utility function in define neural networks

Note:
"""
import torch
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



def create_orth_conv2d( m, dropout ):
    """Create orthogonal conv2d from orginal conv2d
    :param m a conv2d definition
    """
    ichnls, ochnls = m.in_channels, m.out_channels
    sz_kern, stride, padding = m.kernel_size, m.stride, m.padding
    m_bias = m.bias
    print( 'Bias: ', m_bias )
    conv2d_U = torch.nn.Conv2d( ichnls, ochnls, sz_kern, stride, padding, bias=False  )
    conv2d_V = torch.nn.Conv2d( ochnls, ochnls, 1, 1, 0, bias=m_bias )
    conv2d_S = torch.nn.Conv2d( ochnls, ochnls, 1, 1, 0, groups=ochnls, bias=False )

    return [ conv2d_U, conv2d_S, conv2d_V ]


def convert_to_orth_model( model, dropout ):
    """Convert normal model to model with orthogonal channels
    :param model orginal model definition
    :param dropout channel dropout rate
    
    :return orthogonal model
    """
    model_orth = []
    def __convert_layer( model ):
        for m in model.children():
            if isinstance( m, torch.nn.Sequential ):
                __convert_layer( m )
            else:
                if isinstance( m, torch.nn.Conv2d ):
                    m_orth = create_orth_conv2d( m, dropout )
                    model_orth.extend( m )
                else:
                    model_orth.append( m )


    __convert_layer( model )

    print( torch.nn.Sequential( *model_orth ) )
    quit()
