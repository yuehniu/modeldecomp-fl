import torch
import numpy as np
from utils.nn import BasicBlock_Orth, Conv2d_Orth
from model.resnetcifar import BasicBlock
from utils.meter import cal_acc, AverageMeter


class Server( object ):
    def __init__( self, args, model, train_dl, val_dl, logger ):
        self.args = args
        self.model = model
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.criterion = torch.nn.CrossEntropyLoss()
        self.best_acc = 0.0
        self.logger = logger

    def aggregate( self, clients ):
        self.logger.info( 'Server: \taggregate parameter from clients' )
        if self.args.aggr_mode == 'FedAVG':
            self.aggregate_fedavg( clients )
        else:
            raise NotImplementedError

    def create_sub_model( self, clients ):
        """
        create a sub model for each client
        Args:
            clients: a list of activate clients
        Returns:

        Note:
            In this version, we simulate the sub model creation by apply mask to selected channels.
        """
        self.logger.info( 'Server: \tsend aggregated models to clients' )

        for i, client in enumerate( clients ):
            send_sub_model( self.model, client.model )

    def eval( self, r ):
        """
        evaluate the aggregated model on the server side
        Args:
            r: communication round
        Returns:

        """
        avg_loss = AverageMeter()
        avg_acc = AverageMeter()

        self.model.eval()

        for i, ( input, target ) in enumerate( self.val_dl ):
            if self.args.device == 'gpu':
                input = input.cuda()
                target = target.cuda()

            with torch.no_grad():
                output = self.model( input )
                loss = self.criterion( output, target )

            prec1 = cal_acc( output, target )[ 0 ]
            avg_loss.update( loss.item(), input.size( 0 ) )
            avg_acc.update( prec1.item(), input.size( 0 ) )

        if self.best_acc < avg_acc.avg:
            self.best_acc = avg_acc.avg
        self.logger.info('Server: \tRound: [{}]\tLoss {:.4f}\tPrec@1 {:.3f}'.format(
                r, avg_loss.avg, avg_acc.avg
            )
        )

        return avg_acc, avg_loss

    # Aggregate methods

    def aggregate_fedavg( self, clients ):
        # get client parameters, reconstruct, and apply aggregation
        def __recursive_aggregate( module_s, module_c, alpha ):
            for m_s, m_c in zip( module_s.children(), module_c.children() ):
                if isinstance( m_s, torch.nn.Sequential ):
                    assert isinstance( m_c, torch.nn.Sequential ), 'Mismatch between server and client models'
                    __recursive_aggregate( m_s, m_c, alpha )
                else:
                    if isinstance( m_s, BasicBlock ):
                        assert isinstance( m_c, BasicBlock_Orth ) or isinstance( m_c, BasicBlock), \
                            'Mismatch between server and client models'
                        __recursive_aggregate( m_s, m_c, alpha )
                    elif isinstance( m_s, torch.nn.Conv2d ) and isinstance( m_c, Conv2d_Orth ):
                        ichnls, ochnls = m_c.conv2d_V.in_channels, m_c.conv2d_V.out_channels
                        sz_kern = m_c.conv2d_V.kernel_size
                        t_V = m_c.conv2d_V.weight.data.view( ochnls, ichnls * sz_kern[0] * sz_kern[1] )
                        t_s = m_c.conv2d_S.weight.data.view( ochnls, )
                        t_U = m_c.conv2d_U.weight.data.view( ochnls, ochnls )

                        # reconstruct original kernels
                        t_s_norm = torch.norm( t_s )
                        t_s *= m_c.chnl_mask
                        t_s_mask_norm = torch.norm( t_s )
                        scaling = t_s_norm / t_s_mask_norm
                        t_USV = torch.mm( torch.mm( t_U, torch.diag( t_s ) ), t_V )
                        w_USV = t_USV.view( ochnls, ichnls, *sz_kern ) * scaling

                        # apply aggregation
                        m_s.weight.data.add_( w_USV, alpha=alpha )

                        if m_s.bias is not None:
                            m_s.bias.data.add_( m_c.conv2d_U.bias.data, alpha=alpha )
                    elif isinstance( m_s, torch.nn.Conv2d ):
                        assert isinstance( m_c, torch.nn.Conv2d ), 'Mismatch between server and client models'
                        m_s.weight.data.add_( m_c.weight.data, alpha=alpha )
                        if m_s.bias is not None:
                            m_s.bias.data.add_( m_c.bias.data, alpha=alpha )
                    elif isinstance( m_s, torch.nn.BatchNorm2d ):
                        assert isinstance( m_c, torch.nn.BatchNorm2d ), 'Mismatch between server and client models'

                        # upload both parameters and running statistics
                        m_s.weight.data.add_( m_c.weight.data, alpha=alpha )
                        m_s.bias.data.add_( m_c.bias.data, alpha=alpha )
                        m_s.running_mean.data.add_( m_c.running_mean.data, alpha=alpha )
                        m_s.running_var.data.add_( m_c.running_var.data, alpha=alpha )
                    elif isinstance( m_s, torch.nn.Linear ):
                        assert isinstance( m_c, torch.nn.Linear ), 'Mismatch between server and client models'
                        m_s.weight.data.add_( m_c.weight.data, alpha=alpha )
                        if m_s.bias is not None:
                            m_s.bias.data.add_( m_c.bias.data, alpha=alpha )

        # first clear server model parameters and running statistics
        for m in self.model.modules():
            if isinstance( m, torch.nn.Conv2d ) or isinstance( m, torch.nn.Linear ):
                m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance( m, torch.nn.BatchNorm2d ):
                m.weight.data.zero_()
                m.bias.data.zero_()
                m.running_mean.data.zero_()
                m.running_var.data.zero_()

        # apply aggregation
        for client in clients:
            __recursive_aggregate( self.model, client.model, client.alpha )


def send_sub_model( model_s, model_c, random_mask=True ):

    def __recursive_refresh( module_s, module_c ):
        for m_s, m_c in zip( module_s.children(), module_c.children() ):
            if isinstance( m_s, torch.nn.Sequential ):
                assert isinstance( m_c, torch.nn.Sequential ), 'Mismatch between server and client models'
                __recursive_refresh( m_s, m_c )
            else:
                if isinstance( m_s, BasicBlock ):
                    assert isinstance( m_c, BasicBlock_Orth ) or isinstance( m_c, BasicBlock ), \
                        'Mismatch between server and client models'
                    __recursive_refresh( m_s, m_c )
                elif isinstance( m_s, torch.nn.Conv2d ) and isinstance( m_c, Conv2d_Orth ):
                    ichnls, ochnls = m_s.in_channels, m_s.out_channels
                    sz_kern = m_s.kernel_size

                    # decompose kernels
                    t_USV = m_s.weight.data.view( ochnls, ichnls*sz_kern[0]*sz_kern[1] )
                    tt_U, tt_s, tt_V = torch.svd( t_USV )
                    m_c.t_s = tt_s
                    weight_V = tt_V.t().view( ochnls, ichnls, *sz_kern )
                    weight_U = tt_U.view( ochnls, ochnls, 1, 1 )
                    weight_S = tt_s.view( ochnls, 1, 1, 1 )

                    # send to clients
                    m_c.conv2d_V.weight.data.copy_( weight_V )
                    m_c.conv2d_S.weight.data.copy_( weight_S )
                    m_c.conv2d_S.weight.requires_grad = False  # freeze singular values
                    m_c.conv2d_U.weight.data.copy_( weight_U )

                    # generate mask
                    tt_s2 = np.square( tt_s.cpu().numpy() )
                    m_c.p_s = tt_s2 / tt_s2.sum()
                    if random_mask:
                        chnl_keep = np.random.choice( ochnls, m_c.n_keep, replace=False, p=m_c.p_s )
                    else:
                        chnl_keep = np.arange( m_c.n_keep )
                    m_c.chnl_mask = torch.zeros_like( m_c.t_s )
                    m_c.chnl_mask[ chnl_keep ] = 1.0

                    if m_s.bias is not None:
                        m_c.conv2d_U.bias.data.copy_( m_s.bias.data )
                elif isinstance( m_s, torch.nn.Conv2d ):
                    assert isinstance( m_c, torch.nn.Conv2d ), 'Mismatch between server and client models'
                    m_c.weight.data.copy_( m_s.weight.data )
                    if m_s.bias is not None:
                        m_c.bias.data.copy_( m_s.bias.data )
                elif isinstance( m_s, torch.nn.BatchNorm2d ):
                    assert isinstance(m_c, torch.nn.BatchNorm2d), 'Mismatch between server and client models'
                    m_c.weight.data.copy_( m_s.weight.data )
                    m_c.bias.data.copy_( m_s.bias.data )
                    m_c.running_mean.data.copy_( m_s.running_mean )
                    m_c.running_var.data.copy_( m_s.running_var )
                elif isinstance( m_s, torch.nn.Linear ):
                    assert isinstance( m_c, torch.nn.Linear ), 'Mismatch between server and client models'
                    m_c.weight.data.copy_( m_s.weight.data )
                    if m_s.bias is not None:
                        m_c.bias.data.copy_( m_s.bias.data )

    __recursive_refresh( model_s, model_c )

