"""Utility function in define neural networks

Note:
"""
import math
import torch
import copy
import numpy as np
from model.vgg import vgg11, vgg11_bn, vgg16, vgg16_bn, vgg19, vgg19_bn
from model.resnetcifar import resnet18, resnet34, BasicBlock
from model.resnetcifar_tiny import resnet8
from model.femnist import CNN
from model.SentimentRNN import SentimentRNN
from model.deit import deit_tiny_patch16_224

from model.deit import Block, Attention
from timm.models.layers import PatchEmbed, Mlp
from .nn_transformer import Block_Orth, VisionTransformer_Orth


def build_network( model_name, **kwargs ):
    models = {
        'vgg11':    vgg11,
        'vgg11_bn': vgg11_bn,
        'vgg16':    vgg16,
        'vgg16_bn': vgg16_bn,
        'vgg19':    vgg19,
        'vgg19_bn': vgg19_bn,
        'resnet8':  resnet8,
        'resnet18': resnet18,
        'resnet34': resnet34,
        'deit_tiny': deit_tiny_patch16_224
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
            if args.model == 'deit_tiny':
                args.patch = 4
                model = build_network( args.model,pretrained=True,img_size=32,num_classes=10,patch_size=4,args=args )
            else:
                model = build_network( args.model, len_feature=512, num_classes=10 )
        elif args.dataset == 'femnist':
            model = CNN()
        elif args.dataset == 'imdb':
            model = SentimentRNN( vocab_size=vocab_size )
        else:
            raise NotImplementedError

        # convert model if needed
        if args.drop_orthogonal:
            if args.dataset == 'imdb':
                model_s = copy.deepcopy( model )
            else:
                if args.model == 'deit_tiny':
                    blocks_orth = convert_to_orth_model( model.blocks, keep, fl )
                    model_s = VisionTransformer_Orth( model, blocks_orth )
                else:
                    model_s = convert_to_orth_model( model, keep, fl )
        elif args.drop_original:
            if args.dataset == 'imdb':
                model_s = copy.deepcopy( model )
            else:
                model_s = add_original_dropout( model, keep, random=args.random_mask )
        else:
            model_s = copy.deepcopy( model )

        return model_s.to( device )
    else:
        model_c = copy.deepcopy( model )

        # change weight decay in Orth_Conv to Frobenius decay
        layer_with_frodecay = ['conv2d_V', 'conv2d_S', 'conv2d_U', 'fc_U', 'fc_V' ]
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
        t_U_s, t_V_s = t_U * torch.sqrt( t_s ), t_V * torch.sqrt( t_s )
        mask = torch.ones_like( t_s )
        weight_V, weight_U    = t_V_s.t().view( v_ochnls, ichnls, *sz_kern ), t_U_s.view( ochnls, v_ochnls, 1, 1 )
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
        # self.n_keep2 = math.ceil( self.keep * ochnls )
        self.n_keep2 = math.ceil( 0.4 * ochnls )
        self.chnl_mask_v = torch.ones( v_ochnls, device='cuda' )
        self.chnl_left_v = torch.ones( v_ochnls, device='cuda' )
        self.chnl_mask_u = torch.ones( ochnls, v_ochnls, device='cuda' )
        self.chnl_left_u = torch.ones( ochnls, v_ochnls, device='cuda' )
        self.chnl_mask_v_times = torch.zeros( self.v_ochnls, device='cuda' )
        self.chnl_aggr_v_coeff = torch.zeros( self.v_ochnls, device='cuda' )
        self.chnl_mask_u_times = torch.zeros( ochnls, v_ochnls, device='cuda' )
        self.chnl_aggr_u_coeff = torch.zeros( ochnls, v_ochnls, device='cuda' )

        self.t_s = t_s.cuda()
        self.fl, is_decomposed = fl, False

    def forward( self, x ):
        out = self.conv2d_V( x )
        # out = self.conv2d_S( out )
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
                    if m.in_channels == 3 or m.in_channels == 1 or m.kernel_size[0] == 1:
                        # ignore input layer or 1x1 kernel
                        m_orth = copy.deepcopy( m )
                    else:
                        m_orth = Conv2d_Orth_v2( m, keep=keep, fl=fl  )

                # transformer layers
                elif isinstance( m, PatchEmbed ):
                    m_orth = copy.deepcopy( m )
                elif isinstance( m, Block ):
                    m_orth = Block_Orth( m, keep=keep )
                elif isinstance( m, torch.nn.LayerNorm ):
                    m_orth = copy.deepcopy( m )

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

