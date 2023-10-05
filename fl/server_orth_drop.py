import math

import torch
import numpy as np
from utils.nn import BasicBlock_Orth, Conv2d_Orth
from utils.nn_lstm import LSTMs_Orth, LSTM_Orth
from model.resnetcifar import BasicBlock
from utils.meter import cal_entropy


def clear_global_stats(self, model_c):
    def __clear(module, module_c):
        for m, m_c in zip(module.children(), module_c.children()):
            if isinstance( m, torch.nn.Sequential ):
                assert isinstance( m_c, torch.nn.Sequential ), 'Mismatch between server and client models'
                __clear( m, m_c )
            else:
                if isinstance( m, BasicBlock_Orth ) or isinstance( m, BasicBlock ):
                    assert isinstance( m_c, BasicBlock_Orth ) or isinstance( m_c, BasicBlock ), \
                            'Mismatch between server and client models'
                    __clear(m, m_c)
                elif isinstance( m, Conv2d_Orth ) and isinstance( m_c, Conv2d_Orth ):
                    # - keep unselected channels

                    if m.conv2d_U.weight not in self.mm_buffer.keys():
                        self.mm_buffer[ m.conv2d_U.weight ] = torch.zeros_like( m.conv2d_U.weight )
                    if m.conv2d_V.weight not in self.mm_buffer.keys():
                        self.mm_buffer[ m.conv2d_V.weight ] = torch.zeros_like( m.conv2d_V.weight )
                    chnl_left = m.chnl_left.unsqueeze( 1 ).unsqueeze( 1 )
                    w_U_left = m.conv2d_U.weight.data * chnl_left
                    w_V_left = m.conv2d_V.weight.data * chnl_left.unsqueeze( 1 )
                    mm_U_left = self.mm_buffer[ m.conv2d_U.weight ] * chnl_left
                    mm_V_left = self.mm_buffer[ m.conv2d_V.weight ] * chnl_left.unsqueeze(1)

                    m.conv2d_U.weight.data.copy_(w_U_left)
                    m.conv2d_V.weight.data.copy_(w_V_left)
                    self.mm_buffer[m.conv2d_U.weight].data.copy_(mm_U_left)
                    self.mm_buffer[m.conv2d_V.weight].data.copy_(mm_V_left)
                    m.chnl_left = torch.ones_like(m.chnl_left)

                    # - clear all channels
                    """
                    m.conv2d_U.weight.data.zero_()
                    m.conv2d_V.weight.data.zero_()
                    self.mm_buffer[ m.conv2d_U.weight ] = torch.zeros_like( m.conv2d_U.weight )
                    self.mm_buffer[ m.conv2d_V.weight ] = torch.zeros_like( m.conv2d_V.weight )
                    """

                    if m.conv2d_U.bias is not None:
                        m.conv2d_U.bias.data.zero_()
                        self.mm_buffer[m.conv2d_U.bias] = torch.zeros_like(m.conv2d_U.bias)
                elif isinstance(m, torch.nn.Conv2d) and isinstance(m_c, Conv2d_Orth):
                    m.weight.data.zero_()
                    ochnls = m.out_channels
                    self.mm_buffer[m.weight] = [
                        torch.zeros_like(m.weight.data),
                        torch.zeros(ochnls, ochnls, 1, 1, device=torch.device('cuda'))
                    ]
                    if m.bias is not None:
                        m.bias.data.zero_()
                        self.mm_buffer[m.bias] = torch.zeros_like(m.bias.data)
                elif isinstance(m, torch.nn.Conv2d) and isinstance(m_c, torch.nn.Conv2d):
                    m.weight.data.zero_()
                    self.mm_buffer[m.weight] = torch.zeros_like(m.weight.data)
                    if m.bias is not None:
                        m.bias.data.zero_()
                        self.mm_buffer[m.bias] = torch.zeros_like(m.bias.data)
                elif isinstance(m, torch.nn.BatchNorm2d):
                    m.weight.data.zero_()
                    m.bias.data.zero_()
                    # m.running_mean.data.zero_()
                    # m.running_var.data.zero_()
                    self.mm_buffer[m.weight] = torch.zeros_like(m.weight.data)
                    self.mm_buffer[m.bias] = torch.zeros_like(m.bias.data)
                elif isinstance(m, torch.nn.Linear):
                    m.weight.data.zero_()
                    self.mm_buffer[m.weight] = torch.zeros_like(m.weight.data)
                    if m.bias is not None:
                        m.bias.data.zero_()
                        self.mm_buffer[m.bias] = torch.zeros_like(m.bias.data)

                # language model
                elif isinstance( m, torch.nn.Embedding ):
                    m.weight.data.zero_()
                elif isinstance( m, torch.nn.LSTM ):
                    m.weight_ih_l0.data.zero_()
                    m.bias_ih_l0.data.zero_()
                    m.weight_hh_l0.data.zero_()
                    m.bias_hh_l0.data.zero_()
                    m.weight_ih_l1.data.zero_()
                    m.bias_ih_l1.data.zero_()
                    m.weight_hh_l1.data.zero_()
                    m.bias_hh_l1.data.zero_()

    __clear(self.model, model_c)


# Aggregate methods
def aggregate_fedavg(self, clients):
    # get client parameters, reconstruct, and apply aggregation
    def __aggregate(module_s, module_c, alpha, optimizer):
        for m_s, m_c in zip(module_s.children(), module_c.children()):
            if isinstance(m_s, torch.nn.Sequential):
                __aggregate(m_s, m_c, alpha, optimizer)
            else:
                if isinstance(m_s, BasicBlock) or isinstance(m_s, BasicBlock_Orth):
                    __aggregate(m_s, m_c, alpha, optimizer)

                elif isinstance(m_s, Conv2d_Orth) and isinstance(m_c, Conv2d_Orth):
                    w_V_aggr = m_c.conv2d_V.weight * m_s.chnl_aggr_coeff.unsqueeze(1).unsqueeze(1).unsqueeze(1)
                    mm_V = optimizer.state[m_c.conv2d_V.weight]['momentum_buffer'] * \
                           m_s.chnl_aggr_coeff.unsqueeze(1).unsqueeze(1).unsqueeze(1)
                    # w_V_aggr = m_c.conv2d_V.weight
                    # mm_V = optimizer.state[ m_c.conv2d_V.weight ][ 'momentum_buffer' ]
                    m_s.conv2d_V.weight.data.add_(w_V_aggr, alpha=1)
                    self.mm_buffer[m_s.conv2d_V.weight].data.add_(mm_V, alpha=1)

                    w_U_aggr = m_c.conv2d_U.weight * m_s.chnl_aggr_coeff.unsqueeze(1).unsqueeze(1)
                    mm_U = optimizer.state[m_c.conv2d_U.weight]['momentum_buffer'] * \
                           m_s.chnl_aggr_coeff.unsqueeze(1).unsqueeze(1)
                    # w_U_aggr = m_c.conv2d_U.weight
                    # mm_U = optimizer.state[ m_c.conv2d_U.weight ][ 'momentum_buffer' ]
                    m_s.conv2d_U.weight.data.add_(w_U_aggr, alpha=1)
                    self.mm_buffer[m_s.conv2d_U.weight].data.add_(mm_U, alpha=1)
                    m_s.is_decomposed = False

                    if m_s.conv2d_U.bias is not None:
                        m_s.conv2d_U.bias.data.add_(m_c.conv2d_U.bias, alpha=alpha)
                        mm_bias = optimizer.state[m_c.conv2d_U.bias]['momentum_buffer']
                        self.mm_buffer[m_s.conv2d_U.bias].data.add_(mm_bias, alpha=alpha)

                elif isinstance(m_s, torch.nn.Conv2d) and isinstance(m_c, Conv2d_Orth):
                    ichnls, ochnls = m_c.conv2d_V.in_channels, m_c.conv2d_V.out_channels
                    sz_kern = m_c.conv2d_V.kernel_size
                    sz_kern2 = sz_kern[0] * sz_kern[1]

                    # - reconstruct original kernels
                    t_V = m_c.conv2d_V.weight.data.view(ochnls, ichnls * sz_kern2)
                    t_s = m_c.conv2d_S.weight.data.view(ochnls, )
                    t_U = m_c.conv2d_U.weight.data.view(ochnls, ochnls)

                    # - decompose models
                    t_s_norm = torch.norm(t_s)
                    t_s *= m_c.chnl_mask
                    t_s_mask_norm = torch.norm(t_s)
                    scaling = t_s_norm / t_s_mask_norm
                    t_USV = t_U @ torch.diag(t_s) @ t_V

                    # - apply aggregation on the module
                    w_USV = t_USV.view(ochnls, ichnls, *sz_kern)
                    m_s.weight.data.add_(w_USV, alpha=alpha)

                    mm_V = optimizer.state[m_c.conv2d_V.weight]['momentum_buffer']
                    self.mm_buffer[m_s.weight][0].data.add_(mm_V, alpha=alpha)
                    mm_U = optimizer.state[m_c.conv2d_U.weight]['momentum_buffer']
                    self.mm_buffer[m_s.weight][1].data.add_(mm_U, alpha=alpha)

                    if m_s.bias is not None:
                        m_s.bias.data.add_(m_c.conv2d_U.bias.data, alpha=alpha)
                        mm_bias = optimizer.state[m_c.conv2d_U.bias]['momentum_buffer']
                        self.mm_buffer[m_s.bias].data.add_(mm_bias, alpha=alpha)

                elif isinstance(m_s, torch.nn.Conv2d) and isinstance(m_c, torch.nn.Conv2d):
                    m_s.weight.data.add_(m_c.weight.data, alpha=alpha)

                    mm_w = optimizer.state[m_c.weight]['momentum_buffer']
                    self.mm_buffer[m_s.weight].data.add_(mm_w, alpha=alpha)

                    if m_s.bias is not None:
                        m_s.bias.data.add_(m_c.bias.data, alpha=alpha)

                        mm_bias = optimizer.state[m_c.bias]['momentum_buffer']
                        self.mm_buffer[m_s.bias].data.add_(mm_bias, alpha=alpha)

                elif isinstance(m_s, torch.nn.BatchNorm2d):
                    assert isinstance(m_c, torch.nn.BatchNorm2d), 'Mismatch between server and client models'

                    # upload both parameters and running statistics
                    m_s.weight.data.add_(m_c.weight.data, alpha=alpha)
                    m_s.bias.data.add_(m_c.bias.data, alpha=alpha)
                    # m_s.running_mean.data.add_( m_c.running_mean.data, alpha=alpha )
                    # m_s.running_var.data.add_( m_c.running_var.data, alpha=alpha )

                    mm_w = optimizer.state[m_c.weight]['momentum_buffer']
                    self.mm_buffer[m_s.weight].data.add_(mm_w, alpha=alpha)
                    mm_bias = optimizer.state[m_c.bias]['momentum_buffer']
                    self.mm_buffer[m_s.bias].data.add_(mm_bias, alpha=alpha)

                elif isinstance(m_s, torch.nn.Linear):
                    assert isinstance(m_c, torch.nn.Linear), 'Mismatch between server and client models'
                    m_s.weight.data.add_(m_c.weight.data, alpha=alpha)

                    mm_w = optimizer.state[m_c.weight]['momentum_buffer']
                    self.mm_buffer[m_s.weight].data.add_(mm_w, alpha=alpha)
                    if m_s.bias is not None:
                        m_s.bias.data.add_(m_c.bias.data, alpha=alpha)

                        mm_bias = optimizer.state[m_c.bias]['momentum_buffer']
                        self.mm_buffer[m_s.bias].data.add_(mm_bias, alpha=alpha)

                # language model
                elif isinstance( m_s, torch.nn.Embedding):
                    m_s.weight.data.copy_( m_c.weight.data )
                elif isinstance(m_s, torch.nn.LSTM):
                    assert isinstance(m_c, torch.nn.LSTM)
                    weight_ih_l0 = m_c.weight_ih_l0.detach().data
                    w_sub = create_subset(
                        weight_ih_l0, self.args.channel_keep, False, self.args.prob_factor
                    )
                    m_s.weight_ih_l0.data.add_( w_sub, alpha=alpha )
                    m_s.bias_ih_l0.data.add_( m_c.bias_ih_l0.detach().data, alpha=alpha )

                    weight_hh_l0 = m_c.weight_hh_l0.detach().data
                    w_sub = create_subset(
                        weight_hh_l0, self.args.channel_keep, False, self.args.prob_factor
                    )
                    m_s.weight_hh_l0.data.add_( w_sub, alpha=alpha )
                    m_s.bias_hh_l0.data.add_( m_c.bias_hh_l0.detach().data, alpha=alpha )

                    weight_ih_l1 = m_c.weight_ih_l1.detach().data
                    w_sub = create_subset(
                        weight_ih_l1, self.args.channel_keep, False, self.args.prob_factor
                    )
                    m_s.weight_ih_l1.data.add_( w_sub, alpha=alpha )
                    m_s.bias_ih_l1.data.add_( m_c.bias_ih_l1.detach().data, alpha=alpha )

                    weight_hh_l1 = m_c.weight_hh_l1.detach().data
                    w_sub = create_subset(
                        weight_hh_l1, self.args.channel_keep, False, self.args.prob_factor
                    )
                    m_s.weight_hh_l1.data.add_( w_sub, alpha=alpha )
                    m_s.bias_hh_l1.data.add_( m_c.bias_hh_l1.detach().data, alpha=alpha )

    # - apply aggregation
    for client in clients:
        __aggregate(self.model, client.model, client.alpha, client.optimizer)


def send_sub_model(self, client, model_s, model_c, random_mask=True):
    def __refresh(module_s, module_c):
        for m_s, m_c in zip(module_s.children(), module_c.children()):
            if isinstance(m_c, torch.nn.Sequential):
                assert isinstance(m_s, torch.nn.Sequential), 'Mismatch between server and client models'
                __refresh(m_s, m_c)
            else:
                if isinstance(m_c, BasicBlock) or isinstance(m_c, BasicBlock_Orth):
                    assert isinstance(m_s, BasicBlock_Orth) or isinstance(m_s, BasicBlock), \
                        'Mismatch between server and client models'
                    __refresh(m_s, m_c)

                elif isinstance(m_c, Conv2d_Orth):
                    v_ichnls, v_ochnls = m_c.conv2d_V.in_channels, m_c.conv2d_V.out_channels
                    u_ichnls, u_ochnls = m_c.conv2d_U.in_channels, m_c.conv2d_U.out_channels
                    sz_kern = m_c.conv2d_V.kernel_size
                    sz_kern2 = sz_kern[0] * sz_kern[1]
                    if isinstance(m_s, Conv2d_Orth):
                        tt_U = m_s.conv2d_U.weight.data.view( u_ochnls, u_ichnls )
                        tt_s = m_s.conv2d_S.weight.data.view( v_ochnls, )
                        tt_V = m_s.conv2d_V.weight.data.view( v_ochnls, v_ichnls * sz_kern2).t()
                    elif isinstance(m_s, torch.nn.Conv2d):
                        t_USV = m_s.weight.data.view( u_ochnls, v_ichnls * sz_kern2 )
                        tt_U, tt_s, tt_V = torch.svd(t_USV)
                    else:
                        raise ValueError

                    # generate mask
                    m_c.t_s = tt_s
                    # tt_s2 = np.square( tt_s.cpu().numpy() )
                    # tt_s2 = tt_s.cpu().numpy()
                    tt_s2 = tt_s.cpu().numpy() ** self.args.prob_factor
                    m_c.p_s, m_c.chnl_mask = tt_s2 / tt_s2.sum(), torch.zeros_like( m_c.t_s )
                    if random_mask:
                        chnl_keep = np.random.choice( v_ochnls, m_c.n_keep, replace=False, p=m_c.p_s)
                    else:
                        chnl_keep = np.arange( m_c.n_keep )
                    m_c.chnl_mask[chnl_keep] = 1.0
                    if isinstance(m_s, Conv2d_Orth):
                        m_s.chnl_left[chnl_keep] = 0.0
                        m_s.chnl_mask_times += m_c.chnl_mask
                        m_s.chnl_aggr_coeff = 1.0 / m_s.chnl_mask_times
                        m_s.chnl_aggr_coeff[m_s.chnl_aggr_coeff == torch.inf] = 0.0

                    # - mask U and V
                    tt_U_mask = tt_U * m_c.chnl_mask
                    # tt_U_mask = tt_U_mask * m_c.chnl_mask.unsqueeze( 1 )
                    tt_V_mask = tt_V * m_c.chnl_mask
                    w_c_U = tt_U_mask.view( u_ochnls, u_ichnls, 1, 1)
                    w_c_V = tt_V_mask.t().view( v_ochnls, v_ichnls, *sz_kern)
                    w_c_S = tt_s.view( v_ochnls, 1, 1, 1)

                    # - send sub models to clients
                    m_c.conv2d_U.weight.data.copy_(w_c_U)
                    m_c.conv2d_V.weight.data.copy_(w_c_V)
                    m_c.conv2d_S.weight.data.copy_(w_c_S)
                    m_c.conv2d_S.weight.requires_grad = False  # freeze singular values
                    # m_c.conv2d_U.weight.requires_grad = False
                    if m_c.conv2d_U.bias is not None:
                        m_c.conv2d_U.bias.data.copy_(
                            m_s.bias.data if isinstance(m_s, torch.nn.Conv2d) else m_s.conv2d_U.bias
                        )

                    # update momentum
                    if 'momentum_buffer' in client.optimizer.state[m_c.conv2d_U.weight].keys():
                        if isinstance(m_s, Conv2d_Orth):
                            mm_V = self.mm_buffer[m_s.conv2d_V.weight]
                            mm_U = self.mm_buffer[m_s.conv2d_U.weight]
                        else:
                            mm_V = self.mm_buffer[m_s.weight][0]
                            mm_U = self.mm_buffer[m_s.weight][1]
                        mm_U_mask = mm_U * m_c.chnl_mask.unsqueeze(1).unsqueeze(1)
                        mm_V_mask = mm_V * m_c.chnl_mask.unsqueeze(1).unsqueeze(1).unsqueeze(1)
                        client.optimizer.state[m_c.conv2d_V.weight]['momentum_buffer'].copy_(mm_V_mask)
                        # mm_U_mask = mm_U_mask * m_c.chnl_mask.unsqueeze( 1 ).unsqueeze( 1 ).unsqueeze( 1 )
                        client.optimizer.state[m_c.conv2d_U.weight]['momentum_buffer'].copy_(mm_U_mask)
                        if m_c.conv2d_U.bias is not None:
                            if isinstance(m_s, Conv2d_Orth):
                                client.optimizer.state[m_c.conv2d_U.bias]['momentum_buffer'].copy_(
                                    self.mm_buffer[m_s.conv2d_U.bias]
                                )
                            else:
                                client.optimizer.state[m_c.conv2d_U.bias]['momentum_buffer'].copy_(
                                    self.mm_buffer[m_s.bias]
                                )

                elif isinstance(m_s, torch.nn.Conv2d):
                    assert isinstance(m_c, torch.nn.Conv2d), 'Mismatch between server and client models'
                    m_c.weight.data.copy_(m_s.weight.data)

                    if m_s.bias is not None:
                        m_c.bias.data.copy_(m_s.bias.data)

                    if 'momentum_buffer' in client.optimizer.state[m_c.weight].keys():
                        client.optimizer.state[m_c.weight]['momentum_buffer'].copy_(
                            self.mm_buffer[m_s.weight][0]
                        )
                        if m_s.bias is not None:
                            client.optimizer.state[m_c.bias]['momentum_buffer'].copy_(
                                self.mm_buffer[m_s.bias]
                            )
                elif isinstance(m_s, torch.nn.BatchNorm2d):
                    assert isinstance(m_c, torch.nn.BatchNorm2d), 'Mismatch between server and client models'
                    m_c.weight.data.copy_(m_s.weight.data)
                    m_c.bias.data.copy_(m_s.bias.data)
                    # m_c.running_mean.data.copy_( m_s.running_mean )
                    # m_c.running_var.data.copy_( m_s.running_var )

                    if 'momentum_buffer' in client.optimizer.state[m_c.weight].keys():
                        client.optimizer.state[m_c.weight]['momentum_buffer'].copy_(
                            self.mm_buffer[m_s.weight]
                        )
                        client.optimizer.state[m_c.bias]['momentum_buffer'].copy_(
                            self.mm_buffer[m_s.bias]
                        )
                elif isinstance(m_s, torch.nn.Linear):
                    assert isinstance(m_c, torch.nn.Linear), 'Mismatch between server and client models'
                    m_c.weight.data.copy_(m_s.weight.data)
                    if m_s.bias is not None:
                        m_c.bias.data.copy_(m_s.bias.data)

                    if 'momentum_buffer' in client.optimizer.state[m_c.weight].keys():
                        client.optimizer.state[m_c.weight]['momentum_buffer'].copy_(
                            self.mm_buffer[m_s.weight]
                        )
                        if m_s.bias is not None:
                            client.optimizer.state[m_c.bias]['momentum_buffer'].copy_(
                                self.mm_buffer[m_s.bias]
                            )

                # language model
                elif isinstance( m_s, torch.nn.Embedding ):
                    m_c.weight.data.copy_( m_s.weight.data )
                elif isinstance( m_s, torch.nn.LSTM ):
                    weight_ih_l0 = m_s.weight_ih_l0.detach().data
                    w_sub = create_subset( weight_ih_l0, self.args.channel_keep, random_mask, self.args.prob_factor  )
                    m_c.weight_ih_l0.data.copy_( w_sub )
                    m_c.bias_ih_l0.data.copy_( m_s.bias_ih_l0.detach().data )
                    weight_hh_l0 = m_s.weight_hh_l0.detach().data
                    w_sub = create_subset( weight_hh_l0, self.args.channel_keep, random_mask, self.args.prob_factor  )
                    m_c.weight_hh_l0.data.copy_( w_sub )
                    m_c.bias_hh_l0.data.copy_( m_s.bias_hh_l0.detach().data )

                    weight_ih_l1 = m_s.weight_ih_l1.detach().data
                    w_sub = create_subset( weight_ih_l1, self.args.channel_keep, random_mask, self.args.prob_factor )
                    m_c.weight_ih_l1.data.copy_( w_sub )
                    m_c.bias_ih_l1.data.copy_( m_s.bias_ih_l1.detach().data)
                    weight_hh_l1 = m_s.weight_hh_l1.detach().data
                    w_sub = create_subset( weight_hh_l1, self.args.channel_keep, random_mask, self.args.prob_factor )
                    m_c.weight_hh_l1.data.copy_( w_sub )
                    m_c.bias_hh_l1.data.copy_( m_s.bias_hh_l1.detach().data )

    __refresh(model_s, model_c)


def decompose( self ):

    def __decompose( module ):
        for m in module.children():
            if isinstance( m, torch.nn.Sequential ):
                __decompose( m )
            elif isinstance( m, BasicBlock_Orth ):
                __decompose( m )
            elif isinstance( m, Conv2d_Orth ):
                v_ichnls, v_ochnls = m.conv2d_V.in_channels, m.conv2d_V.out_channels
                u_ichnls, u_ochnls = m.conv2d_U.in_channels, m.conv2d_U.out_channels
                sz_kern = m.conv2d_V.kernel_size
                sz_kern2 = sz_kern[ 0 ] * sz_kern[ 1 ]
                t_U = m.conv2d_U.weight.data.view( u_ochnls, u_ichnls )
                t_s = m.conv2d_S.weight.data.view( v_ochnls, )
                t_V = m.conv2d_V.weight.data.view( v_ochnls, v_ichnls * sz_kern2 )
                t_USV = t_U @ torch.diag( t_s ) @ t_V
                tt_U, tt_s, tt_V = torch.svd( t_USV )
                w_s_U = tt_U.view( u_ochnls, u_ichnls, 1, 1 )
                w_s_V = tt_V.t().view( v_ochnls, v_ichnls, *sz_kern )
                w_s_S = tt_s.view( v_ochnls, 1, 1, 1)
                m.conv2d_U.weight.data.copy_(w_s_U)
                m.conv2d_V.weight.data.copy_(w_s_V)
                m.conv2d_S.weight.data.copy_(w_s_S)
                m.is_decomposed = True
                m.chnl_mask_times = torch.zeros( v_ochnls, device='cuda' )
                m.chnl_aggr_coeff = torch.zeros( v_ochnls, device='cuda' )

    __decompose( self.model )


def profile_rank( self, r, model_c ):
    global i
    i = 1

    def __profile( module, module_c ):
        global i
        for m, m_c in zip( module.children(), module_c.children() ):
            if isinstance( m_c, torch.nn.Sequential ):
                __profile( m, m_c )
            elif isinstance( m_c, BasicBlock_Orth ):
                __profile( m, m_c )

            elif isinstance( m, Conv2d_Orth ) and isinstance( m_c, Conv2d_Orth ):
                ochnls = m.conv2d_V.out_channels
                m_str = 'Conv2d_Orth' + str( i ) + '-' + str( ochnls )
                s = m.conv2d_S.weight.data.view( ochnls, )
                s_entropy, s_max = cal_entropy( s, ochnls )
                s_rank = 2 ** s_entropy
                self.writer.add_scalar( 'rank/'+m_str, s_rank, r )
                self.writer.add_scalar( 'smax/'+m_str, s_max, r )

                i += 1

            elif isinstance( m, torch.nn.Conv2d ) and isinstance( m_c, Conv2d_Orth ):
                ichnls, ochnls = m_c.conv2d_V.in_channels, m_c.conv2d_V.out_channels
                sz_kern = m_c.conv2d_V.kernel_size
                sz_kern2 = sz_kern[ 0 ] * sz_kern[ 1 ]
                m_str = 'Conv2d_Orth' + str( i ) + '-' + str( ochnls )
                t_USV = m.weight.data.view( ochnls, ichnls * sz_kern2 )
                _, s, _ = torch.svd(t_USV)
                s_entropy, s_max = cal_entropy( s, ochnls )
                s_rank = 2 ** s_entropy
                self.writer.add_scalar( 'rank/'+m_str, s_rank, r )
                self.writer.add_scalar( 'smax/'+m_str, s_max, r )

                i += 1

    __profile( self.model, model_c )


def profile_sampling( self ):
    global i
    i = 1

    def __profile( module ):
        global i
        for m in module.children():
            if isinstance( m, torch.nn.Sequential ):
                __profile( m )
            elif isinstance( m, BasicBlock_Orth ):
                __profile( m )

            elif isinstance( m, Conv2d_Orth ):
                ochnls = m.conv2d_V.out_channels
                m_str = 'Conv2d_Orth' + str( i ) + '-' + str( ochnls )
                # self.writer.add_scalar( 'rank/'+m_str, s_rank, r )
                # self.writer.add_scalar( 'smax/'+m_str, s_max, r )
                if m_str not in self.sampling_stats.keys():
                    self.sampling_stats[ m_str ] = m.chnl_mask_times.cpu().numpy()
                else:
                    self.sampling_stats[ m_str ] += m.chnl_mask_times.cpu().numpy()

                i += 1

    __profile( self.model )


def create_subset( weight, keep, random=True, prob_factor=1.0 ):
    w_U, s, w_V = torch.svd( weight )
    n_s = w_U.size( 1 )
    # u_ochnls, u_ichnls = w_U.size( 0 ), w_V.size( 1 )
    # v_ochnls, v_ichnls = w_V.size( 0 ), w_V.size( 1 )
    n_keep = math.ceil( keep * n_s )

    # - generate mask
    ss = s.cpu().numpy() ** prob_factor
    p_s, chnl_mask = ss / ss.sum(), torch.zeros_like( s )
    if random:
        chnl_keep = np.random.choice( n_s, n_keep, replace=False, p=p_s )
    else:
        chnl_keep = np.arange( n_keep )
    chnl_mask[ chnl_keep ] = 1.0

    # - mask U and V
    w_U_mask = w_U * chnl_mask
    w_V_mask = w_V * chnl_mask

    w_sub = w_U_mask @ torch.diag( s ) @ w_V_mask.t()

    return w_sub
