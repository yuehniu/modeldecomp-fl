"""
    Orthogonal modules in transformer models
"""
import copy
import math
import torch
import torch.nn as nn

from model.deit import Block, Attention, VisionTransformer
from timm.models.layers import PatchEmbed, Mlp


class VisionTransformer_Orth( nn.Module ):
    def __init__( self, model: VisionTransformer, blocks ):
        super( VisionTransformer_Orth, self ).__init__()
        self.num_classes, self.num_features = model.num_classes, model.num_features
        self.num_tokens = model.num_tokens

        self.patch_embed = copy.deepcopy( model.patch_embed )
        self.cls_token, self.dist_token = copy.deepcopy( model.cls_token ), copy.deepcopy( model.dist_token )
        self.pos_embed, self.pos_drop = copy.deepcopy( model.pos_embed ), copy.deepcopy( model.pos_drop )
        self.blocks = blocks
        self.norm = copy.deepcopy( model.norm )
        self.pre_logits = copy.deepcopy( model.pre_logits )
        self.head, self.head_dist = copy.deepcopy( model.head ), copy.deepcopy( model.head_dist )

    def forward_features( self, x ):
        x = self.patch_embed( x )
        cls_token = self.cls_token.expand( x.shape[0], -1, -1 )  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat( ( cls_token, x ), dim=1 )
        else:
            x = torch.cat( ( cls_token, self.dist_token.expand( x.shape[0], -1, -1), x ), dim=1 )
        x = self.pos_drop( x + self.pos_embed )
        x = self.blocks( x )
        x = self.norm( x )
        if self.dist_token is None:
            return self.pre_logits( x[ :, 0 ] )
        else:
            return x[ :, 0 ], x[ :, 1 ]

    def forward( self, x ):
        x = self.forward_features( x )
        if self.head_dist is not None:
            x, x_dist = self.head( x[ 0 ] ), self.head_dist( x[ 1 ] )  # x must be a tuple
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return ( x + x_dist ) / 2
        else:
            x = self.head( x )
        return x

class Block_Orth( nn.Module ):
    def __init__( self, block, keep ):
        super( Block_Orth, self ).__init__()
        self.norm1     = copy.deepcopy( block.norm1 )
        self.attn      = copy.deepcopy( block.attn )
        self.drop_path = copy.deepcopy( block.drop_path )
        self.norm2     = copy.deepcopy( block.norm2 )
        self.mlp_orth  = Mlp_Orth( block.mlp, keep )

    def forward( self, x ):
        out = x + self.drop_path( self.attn( self.norm1( x ) ) )
        out = out + self.mlp_orth( self.norm2( out ) )

        return out


class Attention_Orth( nn.Module ):
    def __init__(self, attn, keep):
        super( Attention_Orth, self ).__init__()
        pass

    def forward( self, x ):
        pass


class Mlp_Orth( nn.Module ):
    def __init__(self, mlp: Mlp, keep):
        super( Mlp_Orth, self ).__init__()
        self.fc1  = Linear_Orth( mlp.fc1, keep )
        # self.fc1 = copy.deepcopy( mlp.fc1 )
        self.act  = copy.deepcopy( mlp.act )
        self.fc2  = Linear_Orth( mlp.fc2, keep )
        # self.fc2 = copy.deepcopy( mlp.fc2 )
        self.drop = copy.deepcopy( mlp.drop )

    def forward( self, x ):
        out = self.act( self.fc1( x ) )
        out = self.fc2( out )
        out = self.drop( out )

        return out


class Linear_Orth( nn.Module ):
    def __init__( self, fc: nn.Linear, keep: float ):
        super( Linear_Orth, self ).__init__()
        self.in_features, self.out_features = fc.in_features, fc.out_features
        self.u_out_features = min( self.in_features, self.out_features )
        self.has_bias = fc.bias is not None

        # - related to sampling
        self.keep = keep
        self.n_keep  = math.ceil( self.keep * self.u_out_features )
        # self.n_keep2 = math.ceil( self.keep * self.out_features )
        self.n_keep2 = self.out_features

        # - create sub layers
        self.fc_U = nn.Linear( self.in_features, self.u_out_features, bias=False )
        self.fc_V = nn.Linear( self.u_out_features,  self.out_features, bias=self.has_bias )
        w = fc.weight.detach().data.t()
        U, s, V = torch.svd( w )
        Us, Vs = U * torch.sqrt( s ), V * torch.sqrt( s )
        self.fc_U.weight.data.copy_( Us.t() ); self.fc_V.weight.data.copy_( Vs )
        if self.has_bias: self.fc_V.bias.data.copy_( fc.bias.detach().data )

        # - generate mask
        self.s = s.cuda(); self. is_decomposed = False
        self.mask_u = torch.ones( self.u_out_features,device='cuda' )
        self.mask_v = torch.ones( self.out_features, self.u_out_features, device='cuda')
        self.left_u = torch.ones( self.u_out_features, device='cuda' )
        self.left_v = torch.ones( self.out_features, self.u_out_features, device='cuda')
        self.mask_v_times = torch.zeros( self.out_features, self.u_out_features, device='cuda' )
        self.mask_u_times = torch.zeros( self.u_out_features, device='cuda' )
        self.aggr_v_coeff = torch.zeros( self.out_features, self.u_out_features, device='cuda' )
        self.aggr_u_coeff = torch.zeros( self.u_out_features, device='cuda' )

    def forward( self, x ):
        out = self.fc_U( x )
        out = self.fc_V( out )

        return out
