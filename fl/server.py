import torch
import numpy as np
from utils.nn import create_model
from utils.meter import cal_acc, cal_acc_binary, AverageMeter, cal_entropy
from fl import server_orig_drop, server_orth_drop, server_orig, server_orth_drop_v2


class Server( object ):
    def __init__( self, args, model, train_dl, val_dl, logger, writer ):
        self.args = args
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.best_acc = 0.0
        self.best_round = 0
        self.logger = logger
        self.writer = writer
        self.r = 0
        self.sampling_stats = {}

        if self.args.dataset == 'imdb':
            self.criterion = torch.nn.BCELoss()
        else:
            self.criterion = torch.nn.CrossEntropyLoss()

        _, self.model = create_model( args, model=model, fl=True, keep=args.channel_keep )
        # self.model = model

        # global momentum
        self.mm_buffer = {}

        if self.args.drop_original:
            self.clear_global_stats = server_orig_drop.clear_global_stats
            self.send_sub_model = server_orig_drop.send_sub_model
            self.aggregator = server_orig_drop.aggregate_fedavg
        elif self.args.drop_orthogonal:
            """no memory efficiency
            self.send_sub_model     = server_orth_drop.send_sub_model
            self.aggregator         = server_orth_drop.aggregate_fedavg
            self.decompose          = server_orth_drop.decompose
            self.clear_global_stats = server_orth_drop.clear_global_stats
            self.profile_rank       = server_orth_drop.profile_rank
            self.profile_sampling   = server_orth_drop.profile_sampling
            """

            # with memory efficiency
            self.send_sub_model     = server_orth_drop_v2.send_sub_model
            self.aggregator         = server_orth_drop_v2.aggregate_fedavg
            self.decompose          = server_orth_drop_v2.decompose
            self.clear_global_stats = server_orth_drop_v2.clear_global_stats
            self.profile_rank       = server_orth_drop_v2.profile_rank
            self.profile_sampling   = server_orth_drop_v2.profile_sampling
        else:
            self.clear_global_stats = server_orig.clear_global_stats
            self.send_sub_model = server_orig.send_sub_model
            self.aggregator = server_orig.aggregate_fedavg

    def aggregate( self, clients ):
        self.logger.info( 'Server: \taggregate parameter from clients' )
        self.aggregator( self, clients )

    def create_sub_model( self, clients, random_mask=False ):
        """
        create a sub model for each client
        Args:
            clients: a list of activate clients
            random_mask: whether to apply random dropout
        Returns:

        Note:
            In this version, we simulate the sub model creation by apply mask to selected channels.
        """
        self.logger.info( 'Server: \tsend aggregated models to clients' )

        keep = []
        for i, client in enumerate( clients ):
            self.send_sub_model( self, client, self.model, client.model, random_mask=random_mask )
            keep.append( client.channel_keep )
        # self.writer.add_histogram( 'model/keep', np.array( keep ), self.r  )
        # self.profile_sampling( self )

        # - clear server model parameters and running statistics
        self.clear_global_stats( self, clients[ 0 ].model )

    def eval( self, r ):
        """
        evaluate the aggregated model on the server side
        Args:
            r: communication round
        Returns:

        """
        self.r = r
        avg_loss = AverageMeter()
        avg_acc = AverageMeter()

        self.model.eval()

        if self.args.dataset == 'imdb':
            h = self.model.init_hidden( self.args.local_bs )
        for i, ( input, target ) in enumerate( self.val_dl ):
            if self.args.device == 'gpu':
                input = input.cuda()
                target = target.cuda()

            with torch.no_grad():
                if self.args.dataset == 'imdb':
                    h = tuple( [ each.data for each in h ] )
                    # h = [ tuple( [ each.data for each in hi ] ) for hi in h ]
                    output, h = self.model( input, h )
                    loss = self.criterion( output, target.float() )
                else:
                    output = self.model( input )
                    loss = self.criterion( output, target )

            if self.args.dataset == 'imdb':
                prec1 = cal_acc_binary( output, target )
            else:
                prec1 = cal_acc( output, target )[ 0 ]
            avg_loss.update( loss.item(), input.size( 0 ) )
            avg_acc.update( prec1.item(), input.size( 0 ) )

        if self.best_acc < avg_acc.avg:
            self.best_acc = avg_acc.avg
            self.best_round = r
        self.logger.info('Server: \tRound: [{}]\tLoss {:.4f}\tPrec@1 {:.3f}'.format(
                r, avg_loss.avg, avg_acc.avg
            )
        )

        return avg_acc, avg_loss