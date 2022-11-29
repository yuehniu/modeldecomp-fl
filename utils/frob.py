import torch
from utils.nn import BasicBlock_Orth, Conv2d_Orth, Conv2d_Orth_v2
from utils.nn_transformer import Block_Orth, Mlp_Orth, Linear_Orth


def add_frob_decay( model, alpha=0.0001 ):
    """
    Add Frobenius decay to Conv_U/S/V layers
    Args:
        model: nn model definition
        alpha: decay coefficient

    Returns:

    """
    def __recursive_add_decay( module ):
        for m in module.children():
            if isinstance( m, torch.nn.Sequential ):
                __recursive_add_decay( m )
            else:
                if isinstance( m, BasicBlock_Orth ):
                    __recursive_add_decay( m )
                elif isinstance( m, Conv2d_Orth ):
                    # compute Frobenius decay
                    v_ichnls, v_ochnls = m.conv2d_V.in_channels, m.conv2d_V.out_channels
                    u_ichnls, u_ochnls = m.conv2d_U.in_channels, m.conv2d_U.out_channels
                    sz_kern = m.conv2d_V.kernel_size
                    m_V = m.conv2d_V.weight.data.view( v_ochnls, v_ichnls * sz_kern[0] * sz_kern[1] )
                    m_V *= m.chnl_mask.unsqueeze( 1 )  # only perform decay non-masked channels
                    m_s = m.conv2d_S.weight.data.view( v_ochnls, )
                    m_U = m.conv2d_U.weight.data.view( u_ochnls, u_ichnls )
                    m_U *= m.chnl_mask.unsqueeze( 0 )  # only perform decay non-masked channels

                    US = torch.matmul( m_U, torch.diag( m_s ) )
                    SV = torch.matmul( torch.diag( m_s ), m_V )
                    # US = torch.matmul( m_U, torch.diag( torch.sqrt( m_s ) ) )
                    # SV = torch.matmul( torch.diag( torch.sqrt( m_s ) ), m_V )
                    # US, SV = m_U, m_V

                    U_frob = torch.linalg.multi_dot( ( m_U, SV, SV.T ) )
                    V_frob = torch.linalg.multi_dot( ( US.T, US, m_V ) )

                    U_grad = U_frob.view( u_ochnls, u_ichnls, 1, 1 )
                    V_grad = V_frob.view( v_ochnls, v_ichnls, *sz_kern )

                    m.conv2d_U.weight.grad += alpha * U_grad
                    m.conv2d_V.weight.grad += alpha * V_grad

                elif isinstance( m, Conv2d_Orth_v2 ):  # memory efficient version
                    # compute Frobenius decay
                    v_ichnls, v_ochnls = m.conv2d_V.in_channels, m.conv2d_V.out_channels
                    u_ichnls, u_ochnls = m.conv2d_U.in_channels, m.conv2d_U.out_channels
                    sz_kern = m.conv2d_V.kernel_size
                    m_V = m.conv2d_V.weight.data.view( v_ochnls, v_ichnls * sz_kern[0] * sz_kern[1] )
                    m_V *= m.chnl_mask_v.unsqueeze( 1 )  # only perform decay non-masked channels
                    m_s = m.conv2d_S.weight.data.view( v_ochnls, )
                    m_U = m.conv2d_U.weight.data.view( u_ochnls, u_ichnls )
                    m_U *= m.chnl_mask_u  # mask both input and output channels

                    # US = torch.matmul( m_U, torch.diag( m_s ) )
                    # SV = torch.matmul( torch.diag( m_s ), m_V )
                    # US = torch.matmul( m_U, torch.diag( torch.sqrt( m_s ) ) )
                    # SV = torch.matmul( torch.diag( torch.sqrt( m_s ) ), m_V )
                    US, SV = m_U, m_V

                    U_frob = torch.linalg.multi_dot( ( m_U, SV, SV.T ) )
                    V_frob = torch.linalg.multi_dot( ( US.T, US, m_V ) )

                    U_grad = U_frob.view( u_ochnls, u_ichnls, 1, 1 )
                    V_grad = V_frob.view( v_ochnls, v_ichnls, *sz_kern )

                    m.conv2d_U.weight.grad += alpha * U_grad
                    m.conv2d_V.weight.grad += alpha * V_grad

                # transformer layers
                elif isinstance( m, Block_Orth ):
                    __recursive_add_decay( m )
                elif isinstance( m, Mlp_Orth ):
                    __recursive_add_decay( m )
                elif isinstance( m, Linear_Orth ):
                    w_U, w_V = m.fc_U.weight.data, m.fc_V.weight.data
                    w_U *= m.mask_u.unsqueeze( 1 ); w_V *= m.mask_v
                    U_grad = torch.linalg.multi_dot( ( w_U.T, w_V.T, w_V ) )
                    V_grad = torch.linalg.multi_dot( ( w_U, w_U.T, w_V.T ) )

                    m.fc_U.weight.grad += alpha * U_grad.T
                    m.fc_V.weight.grad += alpha * V_grad.T



    __recursive_add_decay( model )
