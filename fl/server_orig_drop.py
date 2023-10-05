import math
import torch
import numpy as np
from utils.nn import BasicBlock_with_dropout, Conv2d_with_dropout
from model.resnetcifar import BasicBlock


def clear_global_stats( self, model_c ):
    def __clear(module, module_c):
        for m, m_c in zip(module.children(), module_c.children()):
            if isinstance(m, torch.nn.Sequential):
                assert isinstance(m_c, torch.nn.Sequential), 'Mismatch between server and client models'
                __clear(m, m_c)
            else:
                if isinstance(m, BasicBlock) or isinstance(m, BasicBlock_with_dropout):
                    assert isinstance(m_c, BasicBlock) or isinstance(m_c, BasicBlock_with_dropout), \
                        'Mismatch between server and client models'
                    __clear(m, m_c)
                elif isinstance( m, Conv2d_with_dropout ) and isinstance( m, Conv2d_with_dropout ):
                    m.conv.weight.data.zero_()
                    self.mm_buffer[m.conv.weight] = torch.zeros_like(m.conv.weight.data)
                    if m.conv.bias is not None:
                        m.conv.bias.data.zero_()
                        self.mm_buffer[m.conv.bias] = torch.zeros_like(m.conv.bias.data)
                elif isinstance( m, torch.nn.Conv2d ) and isinstance( m_c, torch.nn.Conv2d ):
                    m.weight.data.zero_()
                    self.mm_buffer[m.weight] = torch.zeros_like(m.weight.data)
                    if m.bias is not None:
                        m.bias.data.zero_()
                        self.mm_buffer[m.bias] = torch.zeros_like(m.bias.data)
                elif isinstance( m, torch.nn.BatchNorm2d ):
                    m.weight.data.zero_()
                    m.bias.data.zero_()
                    # m.running_mean.data.zero_()
                    # m.running_var.data.zero_()
                    self.mm_buffer[ m.weight ] = torch.zeros_like( m.weight.data )
                    self.mm_buffer[ m.bias ] = torch.zeros_like( m.bias.data )
                elif isinstance( m, torch.nn.Linear ):
                    m.weight.data.zero_()
                    self.mm_buffer[m.weight] = torch.zeros_like( m.weight.data )
                    if m.bias is not None:
                        m.bias.data.zero_()
                        self.mm_buffer[ m.bias ] = torch.zeros_like( m.bias.data )

                # language model
                elif isinstance(m, torch.nn.Embedding):
                    m.weight.data.zero_()
                elif isinstance(m, torch.nn.LSTM):
                    m.weight_ih_l0.data.zero_()
                    m.bias_ih_l0.data.zero_()
                    m.weight_hh_l0.data.zero_()
                    m.bias_hh_l0.data.zero_()
                    m.weight_ih_l1.data.zero_()
                    m.bias_ih_l1.data.zero_()
                    m.weight_hh_l1.data.zero_()
                    m.bias_hh_l1.data.zero_()

    __clear( self.model, model_c )


def aggregate_fedavg( self, clients ):
    # get client parameters, reconstruct, and apply aggregation
    def __aggregate(module_s, module_c, alpha, optimizer):
        for m_s, m_c in zip(module_s.children(), module_c.children()):
            if isinstance(m_s, torch.nn.Sequential):
                __aggregate(m_s, m_c, alpha, optimizer)
            else:
                if isinstance(m_s, BasicBlock) or isinstance(m_s, BasicBlock_with_dropout):
                    __aggregate(m_s, m_c, alpha, optimizer)

                elif isinstance( m_s, Conv2d_with_dropout ) and isinstance( m_c, Conv2d_with_dropout ):
                    W_aggr = m_c.conv.weight.detach().data * m_c.mask.unsqueeze(1).unsqueeze(1).unsqueeze(1)
                    mm_W = optimizer.state[m_c.conv.weight]['momentum_buffer']
                    m_s.conv.weight.data.add_(W_aggr, alpha=alpha)
                    self.mm_buffer[m_s.conv.weight].data.add_(mm_W, alpha=alpha)
                    if m_s.conv.bias is not None:
                        m_s.conv.bias.data.add_(m_c.conv.bias.detach().data, alpha=alpha)
                        mm_bias = optimizer.state[m_c.conv.bias]['momentum_buffer']
                        self.mm_buffer[m_s.conv.bias].data.add_(mm_bias, alpha=alpha)

                elif isinstance(m_s, torch.nn.Conv2d) and isinstance(m_c, torch.nn.Conv2d):
                    m_s.weight.data.add_(m_c.weight.detach().data, alpha=alpha)
                    mm_w = optimizer.state[m_c.weight]['momentum_buffer']
                    self.mm_buffer[m_s.weight].data.add_(mm_w, alpha=alpha)

                    if m_s.bias is not None:
                        m_s.bias.data.add_(m_c.bias.detach().data, alpha=alpha)
                        mm_bias = optimizer.state[m_c.bias]['momentum_buffer']
                        self.mm_buffer[m_s.bias].data.add_(mm_bias, alpha=alpha)

                elif isinstance(m_s, torch.nn.BatchNorm2d):
                    assert isinstance(m_c, torch.nn.BatchNorm2d), 'Mismatch between server and client models'

                    # upload both parameters and running statistics
                    m_s.weight.data.add_(m_c.weight.detach().data, alpha=alpha)
                    m_s.bias.data.add_(m_c.bias.detach().data, alpha=alpha)
                    # m_s.running_mean.data.add_( m_c.running_mean.data, alpha=alpha )
                    # m_s.running_var.data.add_( m_c.running_var.data, alpha=alpha )

                    mm_w = optimizer.state[m_c.weight]['momentum_buffer']
                    self.mm_buffer[m_s.weight].data.add_(mm_w, alpha=alpha)
                    mm_bias = optimizer.state[m_c.bias]['momentum_buffer']
                    self.mm_buffer[m_s.bias].data.add_(mm_bias, alpha=alpha)

                elif isinstance(m_s, torch.nn.Linear):
                    assert isinstance(m_c, torch.nn.Linear), 'Mismatch between server and client models'
                    m_s.weight.data.add_(m_c.weight.detach().data, alpha=alpha)
                    mm_w = optimizer.state[m_c.weight]['momentum_buffer']
                    self.mm_buffer[m_s.weight].data.add_(mm_w, alpha=alpha)
                    if m_s.bias is not None:
                        m_s.bias.data.add_(m_c.bias.detach().data, alpha=alpha)
                        mm_bias = optimizer.state[m_c.bias]['momentum_buffer']
                        self.mm_buffer[m_s.bias].data.add_(mm_bias, alpha=alpha)

                # language model
                elif isinstance(m_s, torch.nn.Embedding):
                    m_s.weight.data.copy_(m_c.weight.data)
                elif isinstance(m_s, torch.nn.LSTM):
                    assert isinstance(m_c, torch.nn.LSTM)
                    weight_ih_l0 = m_c.weight_ih_l0.detach().data
                    w_sub = create_subset(
                        weight_ih_l0, self.args.channel_keep,
                    )
                    m_s.weight_ih_l0.data.add_( w_sub, alpha=alpha )
                    m_s.bias_ih_l0.data.add_( m_c.bias_ih_l0.detach().data, alpha=alpha )

                    weight_hh_l0 = m_c.weight_hh_l0.detach().data
                    w_sub = create_subset(
                        weight_hh_l0, self.args.channel_keep
                    )
                    m_s.weight_hh_l0.data.add_( w_sub, alpha=alpha )
                    m_s.bias_hh_l0.data.add_( m_c.bias_hh_l0.detach().data, alpha=alpha )

                    weight_ih_l1 = m_c.weight_ih_l1.detach().data
                    w_sub = create_subset(
                        weight_ih_l1, self.args.channel_keep
                    )
                    m_s.weight_ih_l1.data.add_( w_sub, alpha=alpha )
                    m_s.bias_ih_l1.data.add_( m_c.bias_ih_l1.detach().data, alpha=alpha )

                    weight_hh_l1 = m_c.weight_hh_l1.detach().data
                    w_sub = create_subset(
                        weight_hh_l1, self.args.channel_keep
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
                if isinstance(m_c, BasicBlock) or isinstance(m_c, BasicBlock_with_dropout):
                    assert isinstance(m_s, BasicBlock_with_dropout) or isinstance(m_s, BasicBlock), \
                        'Mismatch between server and client models'
                    __refresh(m_s, m_c)

                elif isinstance(m_s, Conv2d_with_dropout):
                    assert isinstance(m_c, Conv2d_with_dropout), 'Mismatch between server and client models'
                    W_mask = m_s.conv.weight.detach().data * m_c.mask.unsqueeze(1).unsqueeze(1).unsqueeze(1)
                    m_c.conv.weight.data.copy_(W_mask)
                    if m_s.conv.bias is not None:
                        m_c.conv.bias.data.copy_(m_s.conv.bias.detach().data)
                    """
                    if 'momentum_buffer' in client.optimizer.state[m_c.conv.weight].keys():
                        client.optimizer.state[m_c.conv.weight]['momentum_buffer'].copy_(
                            self.mm_buffer[m_s.conv.weight] * m_c.mask.unsqueeze(1).unsqueeze(1).unsqueeze(1)
                        )
                        if m_s.conv.bias is not None:
                            client.optimizer.state[m_c.conv.bias]['momentum_buffer'].copy_(
                                self.mm_buffer[m_s.conv.bias]
                            )
                    """

                elif isinstance(m_s, torch.nn.Conv2d):
                    assert isinstance(m_c, torch.nn.Conv2d), 'Mismatch between server and client models'
                    m_c.weight.data.copy_(m_s.weight.detach().data)

                    if m_s.bias is not None:
                        m_c.bias.data.copy_(m_s.bias.detach().data)

                    """
                    if 'momentum_buffer' in client.optimizer.state[m_c.weight].keys():
                        client.optimizer.state[m_c.weight]['momentum_buffer'].copy_(
                            self.mm_buffer[m_s.weight][0]
                        )
                    if m_s.bias is not None:
                            client.optimizer.state[m_c.bias]['momentum_buffer'].copy_(
                                self.mm_buffer[m_s.bias]
                            )
                    """
                elif isinstance(m_s, torch.nn.BatchNorm2d):
                    assert isinstance(m_c, torch.nn.BatchNorm2d), 'Mismatch between server and client models'
                    m_c.weight.data.copy_(m_s.weight.detach().data)
                    m_c.bias.data.copy_(m_s.bias.detach().data)
                    # m_c.running_mean.data.copy_( m_s.running_mean )
                    # m_c.running_var.data.copy_( m_s.running_var )

                    """
                    if 'momentum_buffer' in client.optimizer.state[m_c.weight].keys():
                        client.optimizer.state[m_c.weight]['momentum_buffer'].copy_(
                            self.mm_buffer[m_s.weight]
                        )
                        client.optimizer.state[m_c.bias]['momentum_buffer'].copy_(
                            self.mm_buffer[m_s.bias]
                        )
                    """
                elif isinstance(m_s, torch.nn.Linear):
                    assert isinstance(m_c, torch.nn.Linear), 'Mismatch between server and client models'
                    m_c.weight.data.copy_(m_s.weight.detach().data)
                    if m_s.bias is not None:
                        m_c.bias.data.copy_(m_s.bias.detach().data)

                    """
                    if 'momentum_buffer' in client.optimizer.state[m_c.weight].keys():
                        client.optimizer.state[m_c.weight]['momentum_buffer'].copy_(
                            self.mm_buffer[m_s.weight]
                        )
                        if m_s.bias is not None:
                            client.optimizer.state[m_c.bias]['momentum_buffer'].copy_(
                                self.mm_buffer[m_s.bias]
                            )
                    """

                    # language model
                elif isinstance(m_s, torch.nn.Embedding):
                    m_c.weight.data.copy_(m_s.weight.data)
                elif isinstance(m_s, torch.nn.LSTM):
                    weight_ih_l0 = m_s.weight_ih_l0.detach().data
                    w_sub = create_subset( weight_ih_l0, self.args.channel_keep )
                    m_c.weight_ih_l0.data.copy_( w_sub )
                    m_c.bias_ih_l0.data.copy_( m_s.bias_ih_l0.detach().data )
                    weight_hh_l0 = m_s.weight_hh_l0.detach().data
                    w_sub = create_subset( weight_hh_l0, self.args.channel_keep )
                    m_c.weight_hh_l0.data.copy_( w_sub )
                    m_c.bias_hh_l0.data.copy_( m_s.bias_hh_l0.detach().data )

                    weight_ih_l1 = m_s.weight_ih_l1.detach().data
                    w_sub = create_subset( weight_ih_l1, self.args.channel_keep )
                    m_c.weight_ih_l1.data.copy_( w_sub )
                    m_c.bias_ih_l1.data.copy_( m_s.bias_ih_l1.detach().data )
                    weight_hh_l1 = m_s.weight_hh_l1.detach().data
                    w_sub = create_subset( weight_hh_l1, self.args.channel_keep )
                    m_c.weight_hh_l1.data.copy_( w_sub )
                    m_c.bias_hh_l1.data.copy_( m_s.bias_hh_l1.detach().data )

    __refresh(model_s, model_c)


def create_subset( weight, keep ):
    device = torch.device( 'cuda' )
    ( n_ochnls_4, n_ichnls ) = weight.shape
    n_ochnls_1 = n_ochnls_4 // 4
    n_keep = math.ceil( keep * n_ochnls_1 )

    # - generate mask
    chnl_mask = torch.zeros( n_ochnls_4, 1, device=device )
    chnl_mask[ 0:n_keep ] = 1.0
    chnl_mask[ n_ochnls_1:n_ochnls_1 + n_keep ] = 1.0
    chnl_mask[ 2*n_ochnls_1:2*n_ochnls_1 + n_keep ] = 1.0
    chnl_mask[ 3*n_ochnls_1:3*n_ochnls_1 + n_keep ] = 1.0

    # - mask U and V
    w_mask = weight * chnl_mask
    print( w_mask )
    quit()

    return w_mask