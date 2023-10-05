"""
A wrapper for a FL setting up and environment.
"""
import time
import torch
import numpy as np

from data.dataset_fl import create_dataset_fl
from fl.clients import Client
from fl.server import Server
from utils.nn import build_network, convert_to_orth_model, create_model
from utils.meter import AverageMeter


class Context():
    def __init__( self, args, logger, writer ):
        self.args = args
        self.logger = logger
        self.writer = writer
        self.logger.info( 'Init a FL context' )

        # define global and local dataset

        self.global_train_dl, self.global_val_dl, self.local_train_dl, self.local_val_dl, vocab = \
            create_dataset_fl( args )

        # define server model
        vocab_size = len( vocab ) + 1 if args.dataset == 'imdb' else None
        self.model = create_model( args, model=None, fl=True, keep=args.channel_keep, vocab_size=vocab_size )

        # init clients
        self.clients = [ None for _ in range( args.n_clients ) ]
        self.init_clients()

        self.active_indices = None
        self.active_clients = None

        # init server
        self.server = None
        self.init_server()

        # train
        self.global_epoch = 0
        self.round = 0

    def init_clients( self ):
        for i in range( self.args.n_clients ):
            # alpha = len( self.local_train_dl[ i ] ) / len( self.global_train_dl )
            alpha = 1 / ( self.args.active_ratio * self.args.n_clients )
            self.clients[ i ] = Client(
                self.args, i,
                self.local_train_dl[ i ], self.local_val_dl[ i ], alpha,
                model=self.model, logger=self.logger
            )

    def connect_active_clients( self ):
        # connect active clients ( simulation )
        n_active_clients = int( self.args.n_clients * self.args.active_ratio )
        self.active_indices = np.random.choice( self.args.n_clients, n_active_clients, replace=False )
        self.active_clients = [ self.clients[ i ] for i in self.active_indices ]

        self.logger.info( 'Clients connected: {}'.format( self.active_indices ) )

    def init_server( self ):
        self.server = Server(
            self.args, self.model,
            self.global_train_dl, self.global_val_dl,
            self.logger, self.writer
        )

    def fl_train( self, writer=None ):
        model_str = 'orthogonal' if self.args.drop_orthogonal is True else 'original'
        time_train, time_svd, time_aggr, time_submodel = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
        for r in range( self.args.total_round ):
            self.logger.info( '\n' )
            self.logger.info( '-' * 80 )
            self.logger.info( 'FL training with {} model in round {}'.format( model_str, r ) )
            self.logger.info( '-' * 80 )

            # connect active clients
            self.connect_active_clients()

            # create sub model
            random_mask = self.args.random_mask and r >= self.args.warmup_round
            time_sub_beg = time.time()
            self.server.create_sub_model( self.active_clients, random_mask=random_mask )
            time_sub_end = time.time()
            time_submodel.update( time_sub_end-time_sub_beg, 1 )

            # local train
            alpha = 1 / len( self.active_clients )
            time_train_beg = time.time()
            for client in self.active_clients:
                client.alpha = alpha
                client.train( cur_round=r  )
            time_train_end = time.time()
            time_train.update( time_train_end-time_train_beg, 1 )

            # aggregate
            time_aggr_beg = time.time()
            self.server.aggregate( self.active_clients )
            time_aggr_end = time.time()
            time_aggr.update( time_aggr_end-time_aggr_beg, 1 )

            if self.args.drop_orthogonal:
                # re-decompose server model
                time_svd_beg = time.time()
                self.server.decompose( self.server )
                time_svd_end = time.time()
                time_svd.update( time_svd_end-time_svd_beg, 1 )

                # profile rank
                self.server.profile_rank( self.server, r=r, model_c=self.active_clients[ 0 ].model )

            # evaluate global model
            fl_acc1, fl_loss = self.server.eval( r=r )

            self.global_epoch += self.args.local_epoch

            if writer:
                writer.add_scalar( 'server/loss', fl_loss.avg, r )
                writer.add_scalar( 'server/acc1', fl_acc1.avg, r )

            if r % 100 == 0 or r == self.args.total_round-1:
                self.logger.info(
                    'FL training with {} model, keep rate {}, round {}, best accuracy {acc1:.3f} '.format(
                        model_str, self.args.channel_keep, self.server.best_round, acc1=self.server.best_acc
                    )
                )
        self.logger.info( 'Time breakdown: submodel: {} local train: {:.3f}, aggregation: {:.3f}, SVD: {:.3f}'.format(
                time_submodel.avg, time_train.avg, time_aggr.avg, time_svd.avg
            )
        )

        np.save( 'sampling_stats.npy', self.server.sampling_stats )
