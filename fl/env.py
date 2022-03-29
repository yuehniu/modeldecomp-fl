"""
A wrapper for a FL setting up and environment.
"""
import torch
import numpy as np

from data.dataset_fl import create_dataset_fl
from fl.clients import Client
from fl.server import Server
from utils.nn import build_network, convert_to_orth_model, create_model


class Context():
    def __init__( self, args, logger ):
        self.args = args
        self.logger = logger
        self.logger.info( 'Init a FL context' )

        # define server model
        self.model_s = create_model( args, model=None, fl=True )

        # define global and local dataset
        self.global_train_dl, self.global_val_dl, self.local_train_dl, self.local_val_dl = create_dataset_fl( args )

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
            alpha = len( self.local_train_dl[ i ] ) / len( self.global_train_dl )
            self.clients[ i ] = Client(
                self.args, i,
                self.local_train_dl[ i ], self.local_val_dl[ i ], alpha,
                model_s=self.model_s, logger=self.logger
            )

    def connect_active_clients( self ):
        # connect active clients ( simulation )
        n_active_clients = int( self.args.n_clients * self.args.active_ratio )
        self.active_indices = np.random.choice( self.args.n_clients, n_active_clients, replace=False )
        self.active_clients = [ self.clients[ i ] for i in self.active_indices ]

        self.logger.info( 'Clients connected: {}'.format( self.active_indices ) )

    def init_server( self ):
        self.server = Server(
            self.args, self.model_s, self.global_train_dl, self.global_val_dl, self.logger
        )

    def fl_train( self, writer=None ):
        model_str = 'orthogonal' if self.args.drop_orthogonal is True else 'original'
        for r in range( self.args.total_round ):
            self.logger.info( '\n' )
            self.logger.info( '-' * 80 )
            self.logger.info( 'FL training with {} model in round {}'.format( model_str, r ) )
            self.logger.info( '-' * 80 )

            # connect active clients
            self.connect_active_clients()

            # create sub model
            self.server.create_sub_model( self.active_clients )

            # local train
            alpha = 1 / len( self.active_clients )
            for client in self.active_clients:
                client.alpha = alpha
                client.train( cur_round=r  )

            # aggregate
            self.server.aggregate( self.active_clients )

            # evaluate global model
            fl_acc1, fl_loss = self.server.eval( r=r )

            self.global_epoch += self.args.local_epoch

            if writer:
                writer.add_scalar( 'fl_val/loss', fl_loss.avg, r )
                writer.add_scalar( 'fl_val/acc1', fl_acc1.avg, r )

        self.logger.info(
            'FL training with {} model with best accuracy {acc1:.3f}'.format(
                model_str, acc1=self.server.best_acc
            )
        )
