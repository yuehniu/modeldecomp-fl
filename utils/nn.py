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
    def __init__(self, m, keep):
        super( Conv2d_Orth, self ).__init__()
        ichnls, ochnls = m.in_channels, m.out_channels
        sz_kern, stride, padding = m.kernel_size, m.stride, m.padding
        has_bias = m.bias is not None

        # save module paramters
        self.keep = keep
        self.ichnls, self.ochnls, self.sz_kern = ichnls, ochnls, m.kernel_size

        # ------------------------------Convert to Orth layers------------------------------ #
        self.conv2d_V = torch.nn.Conv2d( ichnls, ochnls, sz_kern, stride, padding, bias=False  )
        self.conv2d_S = torch.nn.Conv2d( ochnls, ochnls, 1, 1, 0, groups=ochnls, bias=False )
        self.mask_S = torch.nn.Conv2d( ochnls, ochnls, 1, 1, 0, groups=ochnls, bias=False )
        self.conv2d_U = torch.nn.Conv2d( ochnls, ochnls, 1, 1, 0, bias=has_bias )

        # ------------------------------init weight and bias------------------------------ #
        weight_2d = m.weight.data.view( ochnls, -1 )
        t_U, t_s, t_V = torch.svd( weight_2d )
        mask = torch.ones_like( t_s )
        # print( t_U.shape, t_s.shape, t_V.shape )
        weight_V = t_V.t().view( ochnls, ichnls, *sz_kern )
        weight_U = t_U.view( ochnls, ochnls, 1, 1 )
        weight_S = t_s.view( ochnls, 1, 1, 1 )
        weight_mask = mask.view( ochnls, 1, 1, 1 )
        self.conv2d_V.weight.data.copy_( weight_V )
        self.conv2d_S.weight.data.copy_( weight_S )
        # self.conv2d_S.weight.requires_grad = False
        self.mask_S.weight.data.copy_( weight_mask )
        self.mask_S.weight.requires_grad = False
        self.conv2d_U.weight.data.copy_( weight_U )
        if has_bias:
            self.conv2d_U.bias.data.copy_( m.bias )

        # init sampling probability
        self.p_s = np.ones( self.ochnls ) / self.ochnls
        # self.fix_mask = [ 0, 1, 2, 3 ]
        self.n_keep = math.ceil( self.keep * self.ochnls )
        self.chnl_mask = torch.ones( self.ochnls )
        self.scaling = 1.0
        self.t_s = t_s.cuda()

    def forward( self, x ):
        out = self.conv2d_V( x )
        out = self.conv2d_S( out )
        if self.training:
            with torch.no_grad():
                chnl_keep = np.random.choice( self.ochnls, self.n_keep, replace=False, p=self.p_s )
                chnl_mask = torch.zeros( self.ochnls ).cuda()
                chnl_mask[ chnl_keep ] = 1.0
                # chnl_mask[ self.fix_mask ] = 1.0
                weight_mask = self.chnl_mask.view( self.ochnls, 1, 1, 1 )
                self.mask_S.weight.data.copy_( weight_mask )
                self.scaling = torch.norm( self.t_s ) / torch.norm( self.t_s * chnl_mask )
            out = self.mask_S( out ) * self.scaling
        out = self.conv2d_U( out )

        return out


class BasicBlock_Orth( torch.nn.Module ):
    def __init__( self, block, keep ):
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
                m_orth = Conv2d_Orth( m, keep )
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


def convert_to_orth_model( model, keep ):
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
                    m_orth = BasicBlock_Orth( m, keep=keep )
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


def update_orth_channel( model, optimizer, keep=1.0 ):
    """Update orthogonal channels after certain number of updates
    :param model a model with parameters
    :param optimizer include optimizer to update momentum
    :param keep channel keep rate
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
                    # print( m.conv2d_V.weight.shape )
                    # print( optimizer.state[ m.conv2d_V.weight ][ 'momentum_buffer' ].shape )

                    # ----------------------Combine and Redo SVD---------------------- #
                    t_USV = torch.mm( torch.mm( t_U, torch.diag( t_s ) ), t_V )
                    tt_U, tt_s, tt_V = torch.svd( t_USV )
                    m.t_s = tt_s

                    # apply channel dropout
                    tt_s_np = tt_s.cpu().numpy()
                    m.p_s = tt_s_np / tt_s_np.sum()
                    # chnl_keep = np.random.choice( ochnls, m.n_keep, replace=False, p=m.p_s )
                    # m.chnl_mask = torch.zeros_like( tt_s )
                    # m.chnl_mask[ chnl_keep ] = 1.0
                    # tt_s_mask = tt_s * m.chnl_mask
                    # m.scaling = torch.norm( tt_s ) / torch.norm( tt_s_mask )

                    weight_V = tt_V.t().view( ochnls, ichnls, *sz_kern )
                    weight_U = tt_U.view( ochnls, ochnls, 1, 1 )
                    weight_S = tt_s.view( ochnls, 1, 1, 1 )
                    # weight_mask = m.chnl_mask.view( ochnls, 1, 1, 1 )
                    m.conv2d_V.weight.data.copy_( weight_V )
                    m.conv2d_S.weight.data.copy_( weight_S )
                    # m.conv2d_S.weight.requires_grad = False  # freeze singular values
                    # m.mask_S.weight.data.copy_( weight_mask )
                    # m.mask_S.weight.requires_grad = False
                    m.conv2d_U.weight.data.copy_( weight_U )

                    # ----------------------Update momentum in the layer---------------------- #
                    # TODO: make sure momentum conversion is correct
                    """
                    VTV = tt_V.t() @ t_V.t()
                    UUT = t_U @ tt_U.t()
                    mm_U = optimizer.state[ m.conv2d_U.weight ][ 'momentum_buffer' ].clone()
                    mm_U_2d = mm_U.view( ochnls, -1 )
                    mm_V = optimizer.state[ m.conv2d_V.weight ][ 'momentum_buffer' ].clone()
                    mm_V_2d = mm_V.view( ochnls, -1 )

                    mm_U_2d_convert = mm_U_2d @ UUT
                    mm_V_2d_convert = VTV @ mm_V_2d

                    optimizer.state[ m.conv2d_U.weight ][ 'momentum_buffer' ].copy_(
                        mm_U_2d_convert.view( ochnls, ochnls, 1, 1 )
                    )
                    optimizer.state[ m.conv2d_V.weight ][ 'momentum_buffer' ].copy_(
                        mm_V_2d_convert.view( ochnls, ichnls, *sz_kern )
                    )
                    """

    __recursive_update( model )


class Conv2d_with_dropout( torch.nn.Module ):
    def __init__( self, m, drop ):
        super( Conv2d_with_dropout, self ).__init__()

        self.drop = drop
        self.conv = m
        self.dropout2d = torch.nn.Dropout2d( p=drop )

    def forward( self, x ):
        out = self.conv( x )
        out = self.dropout2d( out )

        return out


class BasicBlock_with_dropout( torch.nn.Module ):
    def __init__( self, block, drop ):
        super( BasicBlock_with_dropout, self ).__init__()
        convs = []
        bns = []
        relu = None
        downsample = None
        self.conv1, self.bn1 = None, None
        self.dropout2d = None
        self.relu = None
        self.conv2, self.bn2 = None, None
        self.downsample = None

        for m in block.children():
            if isinstance( m, torch.nn.Conv2d ):
                dropout2d = torch.nn.Dropout2d( p=drop )
                convs.append( m )
            elif isinstance( m, torch.nn.BatchNorm2d ):
                bns.append( m )
            elif isinstance( m, torch.nn.ReLU ):
                relu = m
            elif isinstance( m, torch.nn.Sequential ):
                downsample = []
                for mm in m.children():
                    if isinstance( mm, torch.nn.Conv2d ):
                        downsample.append( mm )
                    elif isinstance( mm, torch.nn.BatchNorm2d ):
                        downsample.append( mm )
                    else:
                        raise ValueError( 'Unexpected module in downsample of BasicBlock!' )
                downsample = torch.nn.Sequential( *downsample )
            else:
                raise ValueError( 'Unexpected module in BasicBlock!' )

        self.conv1, self.bn1 = convs[ 0 ], bns[ 0 ]
        self.dropout2d = dropout2d
        self.relu = relu
        self.conv2, self.bn2 = convs[ 1 ], bns[ 1 ]
        self.downsample = downsample

    def forward( self, x ):
        residual = x

        out = self.conv1( x )
        out = self.dropout2d( out )
        out = self.bn1( out )
        out = self.relu( out )

        out = self.conv2( out )
        out = self.dropout2d( out )
        out = self.bn2( out )

        if self.downsample is not None:
            residual = self.downsample( x )

        out += residual
        out = self.relu( out )

        return out


def add_regular_dropout( model, drop ):
    """Add regular channel dropout after conv layers
    :param model original model definition
    :param drop dropout rate
    :return orthogonal model
    """
    model_with_dropout = []

    def __convert_layer( module, in_sequential=False ):
        module_new = []
        for m in module.children():
            if isinstance( m, torch.nn.Sequential ):
                module_new = __convert_layer( m, in_sequential=True )
                model_with_dropout.append( torch.nn.Sequential( *module_new ) )
            else:
                if isinstance( m, BasicBlock ):
                    m_with_dropout = BasicBlock_with_dropout( m, drop=drop )
                elif isinstance( m, torch.nn.Conv2d ):
                    if m.in_channels == 3 or m.kernel_size[0] == 1:  # ignore input layer or 1x1 kernel
                        m_with_dropout = m
                    else:
                        m_with_dropout = Conv2d_with_dropout( m, drop=drop )
                else:
                    m_with_dropout = m
                if in_sequential:
                    module_new.append( m_with_dropout )
                else:
                    model_with_dropout.append( m_with_dropout )

        return module_new

    __convert_layer( model )

    return torch.nn.Sequential( *model_with_dropout )
