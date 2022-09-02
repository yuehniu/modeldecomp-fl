"""Utility function in define neural networks

Note:
"""
import math
import torch
import copy
import numpy as np
import torch.jit as jit
from typing import List, Tuple
from torch import Tensor
from model.vgg import vgg11, vgg11_bn, vgg16, vgg16_bn, vgg19, vgg19_bn
from model.resnetcifar import BasicBlock, resnet18, resnet34
from model.femnist import CNN
from model.SentimentRNN import SentimentRNN


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


def create_model( args, model=None, fl=False, keep=1.0, vocab_size=1 ):
    """
    create model for server and clients
    Args:
        args: running arguments
        model: server model
        fl: whether create model for fl or centralized training
        keep: channel keep ratio
        vocab_size: vocabulary size in language models

    Returns:
        server or client models
    """
    device = torch.device( 'cuda' ) if args.device == 'gpu' else torch.device( 'cpu' )

    # create model for the server
    if model is None:
        if args.dataset == 'cifar10':
            model_s = build_network( args.model, len_feature=512, num_classes=10 )
        elif args.dataset == 'femnist':
            model_s = CNN()
        elif args.dataset == 'imdb':
            model_s = SentimentRNN( vocab_size=vocab_size )
        else:
            raise NotImplementedError
        return model_s.to( device )

    # create models for clients
    if args.drop_orthogonal:
        if args.dataset == 'imdb':
            model_c = copy.deepcopy( model )
        else:
            model_c = convert_to_orth_model( model, keep, fl )
    elif args.drop_original:
        if args.dataset == 'imdb':
            model_c = copy.deepcopy( model )
        else:
            model_c = add_original_dropout( model, keep, random=args.random_mask )
    else:
        model_c = copy.deepcopy( model )

    # change weight decay in Orth_Conv to Frobenius decay
    layer_with_frodecay = ['conv2d_V', 'conv2d_S', 'conv2d_U']
    grouped_params = [
        {
            'params': [
                p for n, p in model_c.named_parameters()
                if not any( nf in n for nf in layer_with_frodecay )
            ],
            'weight_decay': args.wd
        },
        {
            'params': [
                p for n, p in model_c.named_parameters()
                if any( nf in n for nf in layer_with_frodecay )
            ],
            'weight_decay': 0.0
        }
    ]

    return grouped_params, model_c.to( device )


# ----------------------------------- Orthogonal dropout ----------------------------------- #


class Conv2d_Orth( torch.nn.Module ):
    def __init__(self, m, keep, fl=False ):
        super( Conv2d_Orth, self ).__init__()
        ichnls, ochnls = m.in_channels, m.out_channels
        sz_kern, stride, padding = m.kernel_size, m.stride, m.padding
        has_bias = m.bias is not None

        # save module parameters
        self.keep = keep
        self.ichnls, self.ochnls, self.sz_kern = ichnls, ochnls, m.kernel_size
        self.v_ochnls = min( ichnls * sz_kern[0] * sz_kern[1], ochnls )

        # ------------------------------Convert to Orth layers------------------------------ #
        v_ochnls = self.v_ochnls
        self.conv2d_V = torch.nn.Conv2d( ichnls, v_ochnls, sz_kern, stride, padding, bias=False  )
        self.conv2d_S = torch.nn.Conv2d( v_ochnls, v_ochnls, 1, 1, 0, groups=v_ochnls, bias=False )
        self.mask_S = torch.nn.Conv2d( v_ochnls, v_ochnls, 1, 1, 0, groups=v_ochnls, bias=False )
        self.conv2d_U = torch.nn.Conv2d( v_ochnls, ochnls, 1, 1, 0, bias=has_bias )

        # ------------------------------init weight and bias------------------------------ #
        weight_2d = m.weight.data.view( ochnls, -1 )
        t_U, t_s, t_V = torch.svd( weight_2d )
        mask = torch.ones_like( t_s )
        weight_V = t_V.t().view( v_ochnls, ichnls, *sz_kern )
        weight_U = t_U.view( ochnls, v_ochnls, 1, 1 )
        weight_S = t_s.view( v_ochnls, 1, 1, 1 )
        weight_mask = mask.view( v_ochnls, 1, 1, 1 )
        self.conv2d_V.weight.data.copy_( weight_V )
        self.conv2d_S.weight.data.copy_( weight_S )
        self.conv2d_S.weight.requires_grad = False
        self.mask_S.weight.data.copy_( weight_mask )
        self.mask_S.weight.requires_grad = False
        self.conv2d_U.weight.data.copy_( weight_U )
        # self.conv2d_U.weight.requires_grad = False
        if has_bias:
            self.conv2d_U.bias.data.copy_( m.bias )

        # mask generation
        self.p_s = np.ones( self.v_ochnls ) / self.v_ochnls
        self.n_keep = math.ceil( self.keep * self.v_ochnls )
        self.fix_mask = np.arange( 0, self.n_keep )
        self.chnl_mask = torch.ones( self.v_ochnls, device='cuda' )
        self.chnl_left = torch.ones( self.v_ochnls, device='cuda' )
        self.chnl_mask_times = torch.zeros( self.v_ochnls, device='cuda' )
        self.chnl_aggr_coeff = torch.zeros( self.v_ochnls, device='cuda' )
        self.scaling = 1.0
        self.t_s = t_s.cuda()
        self.random_mask = True
        self.fl = fl
        self.is_decomposed = False

    def forward( self, x ):
        out = self.conv2d_V( x )
        out = self.conv2d_S( out )
        if self.training:
            with torch.no_grad():
                if not self.fl:
                    if self.random_mask:
                        chnl_keep = np.random.choice( self.ochnls, self.n_keep, replace=False, p=self.p_s )
                    else:
                        chnl_keep = np.arange( self.n_keep )
                    self.chnl_mask = torch.zeros( self.ochnls, device='cuda' )
                    self.chnl_mask[ chnl_keep ] = 1.0
                weight_mask = self.chnl_mask.view( self.v_ochnls, 1, 1, 1 )
                self.mask_S.weight.data.copy_( weight_mask )
                self.scaling = torch.norm( self.t_s ) / torch.norm( self.t_s * self.chnl_mask )
            # TODO: whether or not add scaling in FL training
            out = self.mask_S( out )
        out = self.conv2d_U( out )

        return out


class Conv2d_Orth_v2( torch.nn.Module ):
    def __init__(self, m, keep, fl=False ):
        super( Conv2d_Orth_v2, self ).__init__()
        ichnls, ochnls = m.in_channels, m.out_channels
        sz_kern, stride, padding = m.kernel_size, m.stride, m.padding
        has_bias = m.bias is not None

        # - save module paramters
        self.keep = keep
        self.ichnls, self.ochnls, self.sz_kern = ichnls, ochnls, m.kernel_size
        self.v_ochnls = min( ichnls * sz_kern[0] * sz_kern[1], ochnls )

        # - Convert to Orth layers
        v_ochnls = self.v_ochnls
        self.conv2d_V = torch.nn.Conv2d( ichnls, v_ochnls, sz_kern, stride, padding, bias=False  )
        self.conv2d_S = torch.nn.Conv2d( v_ochnls, v_ochnls, 1, 1, 0, groups=v_ochnls, bias=False )
        self.mask_S   = torch.nn.Conv2d( v_ochnls, v_ochnls, 1, 1, 0, groups=v_ochnls, bias=False )
        self.conv2d_U = torch.nn.Conv2d( v_ochnls, ochnls, 1, 1, 0, bias=has_bias )

        # - init weight and bias
        weight_2d = m.weight.data.view( ochnls, -1 )
        t_U, t_s, t_V = torch.svd( weight_2d )
        mask = torch.ones_like( t_s )
        weight_V, weight_U    = t_V.t().view( v_ochnls, ichnls, *sz_kern ), t_U.view( ochnls, v_ochnls, 1, 1 )
        weight_S, weight_mask = t_s.view( v_ochnls, 1, 1, 1 ), mask.view( v_ochnls, 1, 1, 1 )
        self.conv2d_V.weight.data.copy_( weight_V )
        self.conv2d_S.weight.data.copy_( weight_S )
        self.conv2d_S.weight.requires_grad = False
        self.mask_S.weight.data.copy_( weight_mask )
        self.mask_S.weight.requires_grad = False
        self.conv2d_U.weight.data.copy_( weight_U )
        # self.conv2d_U.weight.requires_grad = False
        if has_bias:
            self.conv2d_U.bias.data.copy_( m.bias )

        # - mask generation
        self.p_s     = np.ones( self.v_ochnls ) / self.v_ochnls
        self.n_keep  = math.ceil( self.keep * self.v_ochnls )
        self.n_keep2 = math.ceil( self.keep * ochnls )
        self.chnl_mask   = torch.ones( v_ochnls, device='cuda' )
        self.chnl_left   = torch.ones( v_ochnls, device='cuda' )
        self.chnl_mask_u = torch.ones( ochnls, v_ochnls, device='cuda' )
        self.chnl_left_u = torch.ones( ochnls, v_ochnls, device='cuda' )
        self.chnl_mask_times   = torch.zeros( self.v_ochnls, device='cuda' )
        self.chnl_aggr_coeff   = torch.zeros( self.v_ochnls, device='cuda' )
        self.chnl_mask_u_times = torch.zeros( ochnls, v_ochnls, device='cuda' )
        self.chnl_aggr_u_coeff = torch.zeros( ochnls, v_ochnls, device='cuda' )

        self.t_s = t_s.cuda()
        self.fl, is_decomposed = fl, False

    def forward( self, x ):
        out = self.conv2d_V( x )
        out = self.conv2d_S( out )
        out = self.conv2d_U( out )

        return out


class BasicBlock_Orth( torch.nn.Module ):
    def __init__( self, block, keep, fl=False ):
        super( BasicBlock_Orth, self ).__init__()

        convs_orth, bns, relu = [], [], None
        downsample = None
        self.conv1, self.bn1 = None, None
        self.relu = None
        self.conv2, self.bn2 = None, None
        self.downsample = None

        for m in block.children():
            if isinstance( m, torch.nn.Conv2d ):
                m_orth = Conv2d_Orth_v2( m, keep, fl )
                convs_orth.append( m_orth )
            elif isinstance( m, torch.nn.BatchNorm2d ):
                bns.append( copy.deepcopy( m ) )
            elif isinstance( m, torch.nn.ReLU ):
                relu = copy.deepcopy( m )
            elif isinstance( m, torch.nn.Sequential ):
                downsample = []
                for mm in m.children():
                    if isinstance( mm, torch.nn.Conv2d ):
                        mm_orth = copy.deepcopy( mm )  # 1x1 conv, no need to convert to 'Conv2d_Oth'
                        downsample.append( mm_orth )
                    elif isinstance( mm, torch.nn.BatchNorm2d ):
                        downsample.append( copy.deepcopy( mm ) )
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


def convert_to_orth_model( model, keep, fl=False ):
    """Convert normal model to model with orthogonal channels
    :param model original model definition
    :param keep channel keep ratio
    :param fl fl or centralized setting
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
                    m_orth = BasicBlock_Orth( m, keep=keep, fl=fl )
                elif isinstance( m, torch.nn.Conv2d ):
                    if m.in_channels == 3 or m.kernel_size[0] == 1:  # ignore input layer or 1x1 kernel
                        m_orth = copy.deepcopy( m )
                    else:
                        m_orth = Conv2d_Orth_v2( m, keep=keep, fl=fl  )
                else:
                    m_orth = copy.deepcopy( m )
                if in_sequential:
                    module_new.append( m_orth )
                else:
                    model_orth.append( m_orth )

        return module_new

    __convert_layer( model )

    return torch.nn.Sequential( *model_orth )


def update_orth_channel( model, optimizer, keep=1.0, random_mask=True ):
    """Update orthogonal channels after certain number of updates
    :param model a model with parameters
    :param optimizer include optimizer to update momentum
    :param keep channel keep rate
    :param random_mask whether mask channels in a random way
    :return updated models
    """
    pchannel_model = []
    channel_model = []

    def __recursive_update( module ):
        for m in module.children():
            if isinstance( m, torch.nn.Sequential ):
                __recursive_update( m )
            else:
                if isinstance( m, BasicBlock_Orth ):
                    __recursive_update( m )
                elif isinstance( m, Conv2d_Orth_v2 ):
                    # print( 'Update orthogonal channel in: ', m )
                    ichnls, ochnls = m.conv2d_V.in_channels, m.conv2d_V.out_channels
                    sz_kern = m.conv2d_V.kernel_size
                    t_V = m.conv2d_V.weight.data.view( ochnls, ichnls*sz_kern[ 0 ]*sz_kern[ 1 ] )
                    t_s = m.conv2d_S.weight.data.view( ochnls, )
                    t_U = m.conv2d_U.weight.data.view( ochnls, ochnls )

                    # ----------------------Combine and Redo SVD---------------------- #
                    # t_s = t_s * m.chnl_mask
                    t_USV = torch.mm( torch.mm( t_U, torch.diag( t_s ) ), t_V )
                    tt_U, tt_s, tt_V = torch.svd( t_USV )
                    m.t_s = tt_s

                    # re-generate channel mask
                    tt_s2 = np.square( tt_s.cpu().numpy() )
                    m.p_s = tt_s2 / tt_s2.sum()
                    """
                    if random_mask:
                        chnl_keep = np.random.choice( ochnls, m.n_keep, replace=False, p=m.p_s )
                    else:
                        chnl_keep = np.arange( m.n_keep )
                    m.chnl_mask = torch.zeros_like( tt_s )
                    m.chnl_mask[ chnl_keep ] = 1.0
                    """

                    tt_s1 = tt_s.cpu().numpy()
                    tt_s_normalized = tt_s1 / tt_s1.sum()
                    channel_entropy = -np.log2( np.sum( tt_s_normalized ** 2 ) )
                    pchannel_model.append( np.ceil( 2 ** channel_entropy ).astype( int ) )
                    channel_model.append( ochnls )

                    weight_V = tt_V.t().view( ochnls, ichnls, *sz_kern )
                    weight_U = tt_U.view( ochnls, ochnls, 1, 1 )
                    weight_S = tt_s.view( ochnls, 1, 1, 1 )
                    # weight_mask = m.chnl_mask.view( ochnls, 1, 1, 1 )
                    m.conv2d_V.weight.data.copy_( weight_V )
                    m.conv2d_S.weight.data.copy_( weight_S )
                    m.conv2d_S.weight.requires_grad = False  # freeze singular values
                    # m.mask_S.weight.data.copy_( weight_mask )
                    # m.mask_S.weight.requires_grad = False
                    m.conv2d_U.weight.data.copy_( weight_U )

                    # ----------------------Update momentum in the layer---------------------- #
                    # TODO: make sure momentum conversion is correct
                    """
                    VTV = torch.mm( tt_V.t(), t_V.t()  )
                    UUT = torch.mm( t_U, tt_U.t() )
                    mm_U = optimizer.state[ m.conv2d_U.weight ][ 'momentum_buffer' ].clone()
                    mm_U_2d = mm_U.view( ochnls, -1 )
                    mm_V = optimizer.state[ m.conv2d_V.weight ][ 'momentum_buffer' ].clone()
                    mm_V_2d = mm_V.view( ochnls, -1 )

                    mm_U_2d_convert = torch.mm( mm_U_2d, UUT )
                    mm_V_2d_convert = torch.mm( VTV, mm_V_2d )

                    optimizer.state[ m.conv2d_U.weight ][ 'momentum_buffer' ].copy_(
                        mm_U_2d_convert.view( ochnls, ochnls, 1, 1 )
                    )
                    optimizer.state[ m.conv2d_V.weight ][ 'momentum_buffer' ].copy_(
                        mm_V_2d_convert.view( ochnls, ichnls, *sz_kern )
                    )
                    """

    __recursive_update( model )

    print( 'total channels    : ', channel_model )
    print( 'principal channels: ', pchannel_model )


# ----------------------------------- Original dropout ----------------------------------- #


class Conv2d_with_dropout( torch.nn.Module ):
    def __init__( self, m, keep, random ):
        super( Conv2d_with_dropout, self ).__init__()

        self.keep = keep
        self.conv = m
        if random:
            self.drop2D = torch.nn.Dropout2d( p=1-keep )
        else:
            ochnls = m.out_channels
            self.drop2D = torch.nn.Conv2d( ochnls, ochnls, 1, 1, 0, groups=ochnls, bias=False )
            self.mask, self.n_keep = torch.zeros( ochnls, device='cuda' ), math.ceil( self.keep * ochnls )
            self.mask[ 0:self.n_keep ] = 1.0
            weight_mask = self.mask.view( ochnls, 1, 1, 1 )
            self.drop2D.weight.data.copy_( weight_mask )
            self.drop2D.weight.requires_grad = False

    def forward( self, x ):
        out = self.conv( x )
        out = self.drop2D( out )

        return out


class BasicBlock_with_dropout( torch.nn.Module ):
    def __init__( self, block, keep, random ):
        super( BasicBlock_with_dropout, self ).__init__()
        convs = []
        bns = []
        relu = None
        downsample = None
        self.conv1, self.bn1 = None, None
        self.relu = None
        self.conv2, self.bn2 = None, None
        self.downsample = None

        for m in block.children():
            if isinstance( m, torch.nn.Conv2d ):
                m_drop = Conv2d_with_dropout( copy.deepcopy( m ), keep=keep, random=random )
                convs.append( m_drop )
            elif isinstance( m, torch.nn.BatchNorm2d ):
                bns.append( copy.deepcopy( m ) )
            elif isinstance( m, torch.nn.ReLU ):
                relu = copy.deepcopy( m )
            elif isinstance( m, torch.nn.Sequential ):
                downsample = []
                for mm in m.children():
                    if isinstance( mm, torch.nn.Conv2d ):
                        down_conv = Conv2d_with_dropout( copy.deepcopy( mm ), keep=keep, random=random )
                        downsample.append( down_conv )
                    elif isinstance( mm, torch.nn.BatchNorm2d ):
                        downsample.append( copy.deepcopy( mm ) )
                    else:
                        raise ValueError( 'Unexpected module in downsample of BasicBlock!' )
                downsample = torch.nn.Sequential( *downsample )
            else:
                raise ValueError( 'Unexpected module in BasicBlock!' )

        self.conv1, self.bn1 = convs[ 0 ], bns[ 0 ]
        self.relu = relu
        self.conv2, self.bn2 = convs[ 1 ], bns[ 1 ]
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


def add_regular_dropout( model, keep ):
    """Add regular channel dropout after conv layers
    :param model original model definition
    :param keep rate
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
                    m_with_dropout = BasicBlock_with_dropout( m, keep=keep )
                elif isinstance( m, torch.nn.Conv2d ):
                    if m.in_channels == 3 or m.kernel_size[0] == 1:  # ignore input layer or 1x1 kernel
                        m_with_dropout = copy.deepcopy( m )
                    else:
                        m_with_dropout = Conv2d_with_dropout( copy.deepcopy( m ), keep=keep )
                else:
                    m_with_dropout = copy.deepcopy( m )
                if in_sequential:
                    module_new.append( m_with_dropout )
                else:
                    model_with_dropout.append( m_with_dropout )

        return module_new

    __convert_layer( model )

    return torch.nn.Sequential( *model_with_dropout )


def add_original_dropout( model, keep, random ):
    """Add original channel dropout after conv layers
    :param model original model definition
    :param keep rate
    :param random dropout or not
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
                    m_with_dropout = BasicBlock_with_dropout( m, keep=keep, random=random )
                elif isinstance( m, torch.nn.Conv2d ):
                    if m.in_channels == 3:
                        m_with_dropout = copy.deepcopy( m )
                    else:
                        m_with_dropout = Conv2d_with_dropout( copy.deepcopy( m ), keep=keep, random=random )
                else:
                    m_with_dropout = copy.deepcopy( m )
                if in_sequential:
                    module_new.append( m_with_dropout )
                else:
                    model_with_dropout.append( m_with_dropout )

        return module_new

    __convert_layer( model )

    return torch.nn.Sequential( *model_with_dropout )


# ----------------------------------- Orthogonal dropout (LSTM) ----------------------------------- #
class LSTM_Orth( torch.nn.Module ):
    def __init__( self, input_size, hidden_size ):
        super( LSTM_Orth, self ).__init__()
        self.input_size, self.hidden_size = input_size, hidden_size

        # input gate
        self.ii_gate = torch.nn.Linear( input_size, hidden_size )
        self.hi_gate = torch.nn.Linear( hidden_size, hidden_size )
        self.i_act = torch.nn.Sigmoid()

        # forget gate
        self.if_gate = torch.nn.Linear( input_size, hidden_size )
        self.hf_gate = torch.nn.Linear( hidden_size, hidden_size )
        self.f_act = torch.nn.Sigmoid()

        # cell memory
        self.ig_gate = torch.nn.Linear( input_size, hidden_size )
        self.hg_gate = torch.nn.Linear( hidden_size, hidden_size )
        self.g_act = torch.nn.Tanh()

        # out gate
        self.io_gate = torch.nn.Linear( input_size, hidden_size )
        self.ho_gate = torch.nn.Linear( hidden_size, hidden_size )
        self.o_act = torch.nn.Sigmoid()

        self.act = torch.nn.Tanh()

    def input_gate( self, x, h ):
        x = self.ii_gate( x )
        h = self.hi_gate( h )
        return self.i_act( x + h )

    def forget_gate( self, x, h):
        x = self.if_gate( x )
        h = self.hf_gate( h )
        return self.f_act( x + h )

    def cell_mem( self, i, f, x, h, c_prev ):
        x = self.ig_gate( x )
        h = self.hg_gate( h )

        k = self.g_act( x + h )
        g = k * i

        c = f * c_prev

        c_next = g + c

        return c_next

    def out_gate(self, x, h ):
        x = self.io_gate( x )
        h = self.ho_gate( h )
        return self.o_act( x + h )

    def forward( self, x, tuple_in: tuple[ Tensor, Tensor ] ):
        ( h, c_prev ) = tuple_in

        i = self.input_gate( x, h )

        f = self.forget_gate( x, h )

        c_next = self.cell_mem( i, f, x, h, c_prev )

        o = self.out_gate( x, h )

        h_next = o * self.act( c_next )

        return h_next, c_next


class LSTMs_Orth( torch.nn.Module ):
    def __init__( self, m, keep ):
        super( LSTMs_Orth, self ).__init__()
        self.input_size, self.hidden_size = m.input_size, m.hidden_size
        self.num_layers = m.num_layers

        device = torch.device( 'cuda' )
        self.LSTM1 = LSTM_Orth( self.input_size, self.hidden_size )
        self.LSTM2 = LSTM_Orth( self.hidden_size, self.hidden_size )
        # self.lstm = []
        # self.lstm.append( LSTM_Orth( self.input_size, self.hidden_size ).to( device ) )
        # for l in range( 1, self.num_layers ):
        #     self.lstm.append( LSTM_Orth( self.hidden_size, self.hidden_size ).to( device ) )

    def forward( self, x, hidden_in ):
        hidden_out = []
        lstm_out = []
        ( h1_i, c1_i ) = hidden_in[ 0 ]
        ( h2_i, c2_i ) = hidden_in[ 1 ]
        for i in range( x.size( 1 ) ):
            h1_i, c1_i = self.LSTM1( x[ :, i, : ], ( h1_i, c1_i ) )
            h2_i, c2_i = self.LSTM2( h1_i, ( h2_i, c2_i ) )
            lstm_out += [ h2_i ]

        lstm_out = torch.stack( lstm_out )
        lstm_out = torch.transpose( lstm_out, 0, 1 )
        hidden_out.append( ( h1_i, c1_i ) )
        hidden_out.append( ( h2_i, c2_i ) )

        return lstm_out, hidden_out


class RNN_Orth( torch.nn.Module ):
    def __init__( self, model, keep ):
        super( RNN_Orth, self ).__init__()
        self.n_lstm_layer = model.n_lstm_layer
        self.dim_hidden, self.dim_embed = model.dim_hidden, model.dim_embed
        self.embedding, self.lstm, self.fc, self.sig = None, None, None, None
        for m in model.children():
            if isinstance( m, torch.nn.Embedding ):
                self.embedding = copy.deepcopy( m )
            elif isinstance( m, torch.nn.LSTM ):
                self.lstm = LSTMs_Orth( m, keep )
            elif isinstance( m, torch.nn.Linear ):
                self.fc = copy.deepcopy( m )
            elif isinstance( m, torch.nn.Sigmoid ):
                self.sig = copy.deepcopy( m )

    def forward( self, x, hidden ):
        batch_size = x.size( 0 )
        embeds = self.embedding( x )
        lstm_out, hidden = self.lstm( embeds, hidden )
        lstm_out = lstm_out.contiguous().view( -1, self.dim_hidden )
        out = self.fc( lstm_out )
        sig_out = self.sig( out )
        sig_out = sig_out.view( batch_size, -1 )
        sig_out = sig_out[ :, -1 ]

        return sig_out, hidden

    def init_hidden( self, batch_size ):
        device = torch.device( 'cuda' )
        hidden = []
        for i in range( self.n_lstm_layer ):
            h0 = torch.zeros( batch_size, self.dim_hidden ).to( device )
            c0 = torch.zeros( batch_size, self.dim_hidden ).to( device )
            hidden.append( ( h0, c0 ) )

        return hidden
