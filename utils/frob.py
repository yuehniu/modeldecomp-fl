import torch
from utils.nn import BasicBlock_Orth, Conv2d_Orth


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
                    ichnls, ochnls = m.conv2d_V.in_channels, m.conv2d_V.out_channels
                    sz_kern = m.conv2d_V.kernel_size
                    m_V = m.conv2d_V.weight.data.view(ochnls, ichnls * sz_kern[0] * sz_kern[1])
                    m_s = m.conv2d_S.weight.data.view(ochnls, )
                    m_U = m.conv2d_U.weight.data.view(ochnls, ochnls)

                    US = torch.matmul( m_U, torch.diag( m_s ) )
                    SV = torch.matmul( torch.diag( m_s ), m_V )

                    U_frob = torch.chain_matmul( m_U, SV, SV.T )
                    V_frob = torch.chain_matmul( US.T, US, m_V )

                    U_grad = U_frob.view( ochnls, ochnls, 1, 1 )
                    V_grad = V_frob.view( ochnls, ichnls, *sz_kern )

                    m.conv2d_U.weight.grad += alpha * U_grad
                    m.conv2d_V.weight.grad += alpha * V_grad

    __recursive_add_decay( model )
