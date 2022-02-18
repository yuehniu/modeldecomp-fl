"""Utility function in define neural networks

Note:
"""
import math
import torch
import numpy as np
from model.vgg import vgg11, vgg11_bn, vgg16, vgg16_bn, vgg19, vgg19_bn
from model.resnetcifar import BasicBlock, resnet18, resnet34


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


class Conv2d_Orth( torch.nn.Module ):
    def __init__(self, m):
        super( Conv2d_Orth, self ).__init__()
        ichnls, ochnls = m.in_channels, m.out_channels
        sz_kern, stride, padding = m.kernel_size, m.stride, m.padding
        has_bias = m.bias is not None

        # ------------------------------Convert to Orth layers------------------------------ #
        self.conv2d_V = torch.nn.Conv2d( ichnls, ochnls, sz_kern, stride, padding, bias=False  )
        self.conv2d_S = torch.nn.Conv2d( ochnls, ochnls, 1, 1, 0, groups=ochnls, bias=False )
        self.conv2d_U = torch.nn.Conv2d( ochnls, ochnls, 1, 1, 0, bias=has_bias )

        # ------------------------------init weight and bias------------------------------ #
        weight_2d = m.weight.data.view( ochnls, -1 )
        t_U, t_s, t_V = torch.svd( weight_2d )
        # print( t_U.shape, t_s.shape, t_V.shape )
        weight_V = t_V.t().view( ochnls, ichnls, *sz_kern )
        weight_U = t_U.view( ochnls, ochnls, 1, 1 )
        weight_S = t_s.view( ochnls, 1, 1, 1 )
        self.conv2d_V.weight.data.copy_( weight_V )
        self.conv2d_S.weight.data.copy_( weight_S )
        self.conv2d_U.weight.data.copy_( weight_U )
        if has_bias:
            self.conv2d_U.bias.data.copy_( m.bias )

    def forward( self, x ):
        out = self.conv2d_V( x )
        out = self.conv2d_S( out )
        out = self.conv2d_U( out )

        return out


class BasicBlock_Orth( torch.nn.Module ):
    def __init__( self, block ):
        super( BasicBlock_Orth, self ).__init__()
        convs_orth = []
        bns = []
        relu = None
        downsample = None
        self.conv1, self.bn1 = None, None
        self.relu = None
        self.conv2, self.bn2 = None, None
        self.downsample = None

        for m in block.children():
            if isinstance( m, torch.nn.Conv2d ):
                m_orth = Conv2d_Orth( m )
                convs_orth.append( m_orth )
            elif isinstance( m, torch.nn.BatchNorm2d ):
                bns.append( m )
            elif isinstance( m, torch.nn.ReLU ):
                relu = m
            elif isinstance( m, torch.nn.Sequential ):
                downsample = []
                for mm in m.children():
                    if isinstance( mm, torch.nn.Conv2d ):
                        # mm_orth = Conv2d_Orth( mm )
                        mm_orth = mm  # don't convert to Conv2d_Orth
                        downsample.append( mm_orth )
                    elif isinstance( mm, torch.nn.BatchNorm2d ):
                        downsample.append( mm )
                    else:
                        raise ValueError( 'Unexpected module in downsample of BasicBlock!' )
                downsample = torch.nn.Sequential( *downsample )
            else:
                raise ValueError( 'Unexpected module in BasicBlock!' )

        self.conv1, self.bn1 = convs_orth[ 0 ], bns[ 0 ]
        self.relu = relu
        self.conv2, self.bn2 = convs_orth[ 1 ], bns[ 1 ]
        self.downsample = downsample

    def forward( self, x ):
        residual = x

        out = self.conv1( x )
        out = self.bn1( out )
        out = self.relu( out )

        out = self.conv2( out )
        out = self.bn2( out )

        if self.downsample is not None:
            residual = self.downsample( x )

        out += residual
        out = self.relu( out )

        return out


def convert_to_orth_model( model ):
    """Convert normal model to model with orthogonal channels
    :param model original model definition
    :return orthogonal model
    """
    model_orth = []

    def __convert_layer( module, in_sequential=False ):
        module_new = []
        for m in module.children():
            if isinstance( m, torch.nn.Sequential ):
                module_new = __convert_layer( m, in_sequential=True )
                model_orth.append( torch.nn.Sequential( *module_new ) )
            else:
                if isinstance( m, BasicBlock ):
                    m_orth = BasicBlock_Orth( m )
                elif isinstance( m, torch.nn.Conv2d ):
                    if m.in_channels == 3 or m.kernel_size[0] == 1:  # ignore input layer or 1x1 kernel
                        m_orth = m
                    else:
                        m_orth = Conv2d_Orth( m )
                else:
                    m_orth = m
                if in_sequential:
                    module_new.append( m_orth )
                else:
                    model_orth.append( m_orth )

        return module_new

    __convert_layer( model )

    return torch.nn.Sequential( *model_orth )


def update_orth_channel( model, dropout=1.0 ):
    """Update orthogonal channels after certain number of updates
    :param model a model with parameters
    :param dropout channel dropout rate
    :return updated models
    """
    def __recursive_update( module ):
        for m in module.children():
            if isinstance( m, torch.nn.Sequential ):
                __recursive_update( m )
            else:
                if isinstance( m, BasicBlock_Orth ):
                    __recursive_update( m )
                elif isinstance( m, Conv2d_Orth ):
                    # print( 'Update orthogonal channel in: ', m )
                    ichnls, ochnls = m.conv2d_V.in_channels, m.conv2d_V.out_channels
                    sz_kern = m.conv2d_V.kernel_size
                    t_V = m.conv2d_V.weight.data.view( ochnls, ichnls*sz_kern[ 0 ]*sz_kern[ 1 ] )
                    t_s = m.conv2d_S.weight.data.view( ochnls, )
                    t_U = m.conv2d_U.weight.data.view( ochnls, ochnls )

                    # ----------------------Combine and Redo SVD---------------------- #
                    t_USV = torch.mm( torch.mm( t_U, torch.diag( t_s ) ), t_V )
                    tt_U, tt_s, tt_V = torch.svd( t_USV )

                    # apply channel dropout
                    n_keep = math.ceil( dropout * ochnls )
                    tt_s_np = tt_s.cpu().numpy()
                    p_s = tt_s_np / tt_s_np.sum()
                    chnl_keep = np.random.choice( ochnls, n_keep, replace=False, p=p_s )
                    chnl_mask = torch.zeros_like( tt_s )
                    chnl_mask[ chnl_keep ] = 1.0
                    tt_s = tt_s * chnl_mask

                    weight_V = tt_V.t().view( ochnls, ichnls, *sz_kern )
                    weight_U = tt_U.view( ochnls, ochnls, 1, 1 )
                    weight_S = tt_s.view( ochnls, 1, 1, 1 )
                    m.conv2d_V.weight.data.copy_( weight_V )
                    m.conv2d_S.weight.data.copy_( weight_S )
                    m.conv2d_S.weight.requires_grad = False  # do not update singular values
                    m.conv2d_U.weight.data.copy_( weight_U )

    __recursive_update( model )
