import torch
import numpy as np
from utils.nn import BasicBlock_with_dropout, Conv2d_with_dropout
from model.resnetcifar import BasicBlock

# transformer modules
from model.deit import Block, Attention
from timm.models.layers import PatchEmbed, Mlp


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
                elif isinstance( m, torch.nn.Conv2d ) and isinstance( m_c, torch.nn.Conv2d ):
                    m.weight.data.zero_()
                    self.mm_buffer[m.weight] = torch.zeros_like(m.weight.data)
                    if m.bias is not None:
                        m.bias.data.zero_()
                        self.mm_buffer[m.bias] = torch.zeros_like(m.bias.data)
                elif isinstance( m, torch.nn.BatchNorm2d ):
                    m.weight.data.zero_(); m.bias.data.zero_()
                    # m.running_mean.data.zero_(); m.running_var.data.zero_()
                    self.mm_buffer[ m.weight ] = torch.zeros_like( m.weight.data )
                    self.mm_buffer[ m.bias ]   = torch.zeros_like( m.bias.data )
                elif isinstance( m, torch.nn.Linear ):
                    m.weight.data.zero_()
                    self.mm_buffer[m.weight] = torch.zeros_like( m.weight.data )
                    if m.bias is not None:
                        m.bias.data.zero_()
                        self.mm_buffer[ m.bias ] = torch.zeros_like( m.bias.data )

                # language model
                elif isinstance( m, torch.nn.Embedding ):
                    m.weight.data.zero_()
                elif isinstance( m, torch.nn.LSTM ):
                    m.weight_ih_l0.data.zero_(); m.bias_ih_l0.data.zero_()
                    m.weight_hh_l0.data.zero_(); m.bias_hh_l0.data.zero_()
                    m.weight_ih_l1.data.zero_(); m.bias_ih_l1.data.zero_()
                    m.weight_hh_l1.data.zero_(); m.bias_hh_l1.data.zero_()

                # transformer model
                elif isinstance( m, PatchEmbed ):
                    __clear( m, m_c )
                elif isinstance( m, Block ):
                    __clear( m, m_c )
                elif isinstance( m, Attention ):
                    __clear( m, m_c )
                elif isinstance( m, Mlp ):
                    __clear( m, m_c )
                elif isinstance( m, torch.nn.LayerNorm ):
                    m.weight.data.zero_(); m.bias.data.zero_()
                    self.mm_buffer[ m.weight ] = torch.zeros_like( m.weight.data )
                    self.mm_buffer[ m.bias ]   = torch.zeros_like( m.bias.data   )

                else:
                    pass

    __clear( self.model, model_c )


def aggregate_fedavg( self, clients ):
    # get client parameters, reconstruct, and apply aggregation
    def __aggregate(module_s, module_c, alpha, optimizer):
        for m_s, m_c in zip(module_s.children(), module_c.children()):
            if isinstance(m_s, torch.nn.Sequential):
                __aggregate(m_s, m_c, alpha, optimizer)
            else:
                if isinstance(m_s, BasicBlock):
                    __aggregate(m_s, m_c, alpha, optimizer)

                elif isinstance(m_s, torch.nn.Conv2d):
                    m_s.weight.data.add_( m_c.weight.detach().data, alpha=alpha )
                    mm_w = optimizer.state[ m_c.weight ][ 'momentum_buffer' ]
                    self.mm_buffer[m_s.weight].data.add_(mm_w, alpha=alpha)

                    if m_s.bias is not None:
                        m_s.bias.data.add_( m_c.bias.detach().data, alpha=alpha )
                        mm_bias = optimizer.state[ m_c.bias ][ 'momentum_buffer' ]
                        self.mm_buffer[ m_s.bias ].data.add_( mm_bias, alpha=alpha )

                elif isinstance( m_s, torch.nn.BatchNorm2d ):
                    # upload both parameters and running statistics
                    m_s.weight.data.add_( m_c.weight.detach().data, alpha=alpha )
                    m_s.bias.data.add_( m_c.bias.detach().data, alpha=alpha )
                    # m_s.running_mean.data.add_( m_c.running_mean.data, alpha=alpha )
                    # m_s.running_var.data.add_( m_c.running_var.data, alpha=alpha )

                    mm_w = optimizer.state[ m_c.weight ][ 'momentum_buffer' ]
                    self.mm_buffer[ m_s.weight ].data.add_( mm_w, alpha=alpha )
                    mm_bias = optimizer.state[ m_c.bias ][ 'momentum_buffer' ]
                    self.mm_buffer[ m_s.bias ].data.add_( mm_bias, alpha=alpha )

                elif isinstance( m_s, torch.nn.Linear ):
                    m_s.weight.data.add_( m_c.weight.detach().data, alpha=alpha )
                    mm_w = optimizer.state[ m_c.weight ][ 'momentum_buffer' ]
                    self.mm_buffer[ m_s.weight ].data.add_( mm_w, alpha=alpha )
                    if m_s.bias is not None:
                        m_s.bias.data.add_( m_c.bias.detach().data, alpha=alpha )
                        mm_bias = optimizer.state[ m_c.bias ][ 'momentum_buffer' ]
                        self.mm_buffer[ m_s.bias ].data.add_( mm_bias, alpha=alpha )

                # language model layers
                elif isinstance( m_s, torch.nn.Embedding ):
                    assert isinstance( m_c, torch.nn.Embedding )
                    m_s.weight.data.add_( m_c.weight.detach().data, alpha=alpha )
                elif isinstance( m_s, torch.nn.LSTM ):
                    assert isinstance( m_c, torch.nn.LSTM )
                    m_s.weight_ih_l0.data.add_( m_c.weight_ih_l0.detach().data, alpha=alpha )
                    m_s.bias_ih_l0.data.add_( m_c.bias_ih_l0.detach().data, alpha=alpha )
                    m_s.weight_hh_l0.data.add_( m_c.weight_hh_l0.detach().data, alpha=alpha )
                    m_s.bias_hh_l0.data.add_( m_c.bias_hh_l0.detach().data, alpha=alpha )
                    m_s.weight_ih_l1.data.add_(m_c.weight_ih_l1.detach().data, alpha=alpha)
                    m_s.bias_ih_l1.data.add_(m_c.bias_ih_l1.detach().data, alpha=alpha)
                    m_s.weight_hh_l1.data.add_(m_c.weight_hh_l1.detach().data, alpha=alpha)
                    m_s.bias_hh_l1.data.add_(m_c.bias_hh_l1.detach().data, alpha=alpha)

                # transformer layers
                elif isinstance( m_s, PatchEmbed ):
                    __aggregate( m_s, m_c, alpha, optimizer )
                elif isinstance( m_s, Block ):
                    __aggregate( m_s, m_c, alpha, optimizer )
                elif isinstance( m_s, Attention ):
                    __aggregate( m_s, m_c, alpha, optimizer )
                elif isinstance( m_s, Mlp ):
                    __aggregate( m_s, m_c, alpha, optimizer )
                elif isinstance( m_s, torch.nn.LayerNorm ):
                    m_s.weight.data.add_( m_c.weight.detach().data, alpha=alpha )
                    m_s.bias.data.add_( m_c.bias.detach().data, alpha=alpha )
                    mm_w = optimizer.state[ m_c.weight ][ 'momentum_buffer' ]
                    self.mm_buffer[ m_s.weight ].data.add_( mm_w, alpha=alpha )
                    mm_bias = optimizer.state[ m_c.bias ][ 'momentum_buffer' ]
                    self.mm_buffer[m_s.bias].data.add_( mm_bias, alpha=alpha )

                else:
                    pass

    # - apply aggregation
    for client in clients:
        __aggregate(self.model, client.model, client.alpha, client.optimizer)


def send_sub_model(self, client, model_s, model_c, random_mask=True):

    def __refresh(module_s, module_c):
        for m_s, m_c in zip(module_s.children(), module_c.children()):
            if isinstance(m_c, torch.nn.Sequential):
                assert isinstance(m_s, torch.nn.Sequential), 'Mismatch between server and client models'
                __refresh( m_s, m_c )
            else:
                if isinstance(m_c, BasicBlock) or isinstance(m_c, BasicBlock_with_dropout):
                    __refresh( m_s, m_c )

                elif isinstance(m_s, torch.nn.Conv2d):
                    m_c.weight.data.copy_(m_s.weight.detach().data)

                    if m_s.bias is not None:
                        m_c.bias.data.copy_( m_s.bias.detach().data )

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
                    m_c.weight.data.copy_( m_s.weight.detach().data )
                    m_c.bias.data.copy_( m_s.bias.detach().data )
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
                    m_c.weight.data.copy_( m_s.weight.detach().data )
                    if m_s.bias is not None:
                        m_c.bias.data.copy_( m_s.bias.detach().data )

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

                # language model layer
                elif isinstance( m_s, torch.nn.Embedding ):
                    m_c.weight.data.copy_( m_s.weight.detach().data )
                elif isinstance( m_s, torch.nn.LSTM ):
                    m_c.weight_ih_l0.data.copy_( m_s.weight_ih_l0.detach().data )
                    m_c.bias_ih_l0.data.copy_( m_s.bias_ih_l0.detach().data )
                    m_c.weight_hh_l0.data.copy_( m_s.weight_hh_l0.detach().data )
                    m_c.bias_hh_l0.data.copy_( m_s.bias_hh_l0.detach().data )
                    m_c.weight_ih_l1.data.copy_( m_s.weight_ih_l1.detach().data )
                    m_c.bias_ih_l1.data.copy_( m_s.bias_ih_l1.detach().data )
                    m_c.weight_hh_l1.data.copy_( m_s.weight_hh_l1.detach().data )
                    m_c.bias_hh_l1.data.copy_( m_s.bias_hh_l1.detach().data )

                # transformer layers
                elif isinstance( m_s, PatchEmbed ):
                    __refresh( m_s, m_c )
                elif isinstance( m_s, Block ):
                    __refresh( m_s, m_c )
                elif isinstance( m_s, Attention ):
                    __refresh( m_s, m_c )
                elif isinstance( m_s, Mlp ):
                    __refresh( m_s, m_c )
                elif isinstance( m_s, torch.nn.LayerNorm ):
                    m_c.weight.data.copy_( m_s.weight.detach().data )
                    m_c.bias.data.copy_( m_s.bias.detach().data )

                else:
                    pass

    __refresh(model_s, model_c)
