import time
import math
import torch
import numpy as np

from utils.nn import create_model, Conv2d_Orth, BasicBlock_Orth
from utils.meter import AverageMeter, cal_acc, cal_acc_binary
from utils.frob import add_frob_decay


class Client( object ):
    def __init__( self, args, index, train_dl, val_dl, alpha, model, logger ):
        self.args = args
        self.client_index = index
        self.logger = logger

        self.train_dl, self.val_dl, self.alpha = train_dl, val_dl, alpha

        assert model is not None, 'Define server model first!!!'
        self.channel_keep = args.channel_keep
        if args.hetero:
            self.channel_keep = np.random.choice( args.client_capacities, 1, p=args.client_ratios )[ 0 ]
        self.grouped_params, self.model = create_model( args, model=model, fl=True, keep=self.channel_keep )
        if self.args.dataset == 'imdb':
            self.criterion = torch.nn.BCELoss()
        else:
            self.criterion = torch.nn.CrossEntropyLoss()
        if args.device == 'gpu':
            self.criterion.cuda()

        self.optimizer = torch.optim.SGD(
            self.grouped_params, args.lr,
            momentum=args.momentum, weight_decay=args.wd
        )

        if args.decay == 'cos':
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, args.total_round, eta_min=0.0001
            )
        elif args.decay == 'multisteps':
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=[50, 100, 150], gamma=0.1
            )
        else:
            raise ValueError('Unexpected lr scheduler!!!')

    def train( self, cur_round ):
        """
        train entry point at each local client
        Args:
            cur_round: current communication round
        Returns:
        """
        cur_global_epoch = cur_round * self.args.local_epoch
        cur_local_epoch = 0
        for epoch in range( cur_global_epoch, cur_global_epoch + self.args.local_epoch ):
            self.train_one_epoch( epoch, True )
            # self.eval( epoch )

            self.lr_scheduler.step( epoch=cur_round )

            cur_local_epoch += 1

    def train_one_epoch( self, epoch, display ):
        batch_time, data_time = AverageMeter(), AverageMeter()
        avg_loss, avg_acc = AverageMeter(), AverageMeter()

        self.model.train()

        end = time.time()
        if self.args.dataset == 'imdb':
            h = self.model.init_hidden( self.args.local_bs )
        for i, ( input, target ) in enumerate( self.train_dl ):
            # record data loading time
            data_time.update( time.time() - end )

            if self.args.device == 'gpu':
                input, target = input.cuda(), target.cuda()

            self.optimizer.zero_grad()

            # forward and backward
            if self.args.dataset == 'imdb':
                h = tuple( [ each.data for each in h ] )
                # h = [ tuple( [ each.data for each in hi ] ) for hi in h ]
                output, h = self.model( input, h )
                loss = self.criterion( output, target.float() )
            else:
                output = self.model( input )
                loss = self.criterion( output, target )

            loss.backward()

            # add Frobenius decay
            if self.args.drop_orthogonal:
                add_frob_decay( self.model, alpha=self.args.wd )

            torch.nn.utils.clip_grad_norm_( self.model.parameters(), 5.0 )

            # update parameter
            self.optimizer.step()

            # report acc and loss
            if self.args.dataset == 'imdb':
                prec1 = cal_acc_binary( output, target )
            else:
                prec1 = cal_acc( output, target )[ 0 ]

            avg_acc.update( prec1.item(), input.size( 0 ) )
            avg_loss.update( loss.item(), input.size( 0 ) )

            # record elapsed time
            batch_time.update( time.time() - end )
            end = time.time()

            if i == len( self.train_dl )-1 and display:
                self.logger.info(
                    'Client: {0}\t Epoch: [{1}][{2}/{3}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {avg_loss.avg:.4f}\t'
                    'Prec@1 {avg_acc.avg:.3f}'.format(
                        self.client_index, epoch, i, len( self.train_dl ), batch_time=batch_time,
                        data_time=data_time, avg_loss=avg_loss, avg_acc=avg_acc
                    )
                )

        return avg_acc, avg_loss

    def eval( self, epoch ):
        avg_loss = AverageMeter()
        avg_acc = AverageMeter()

        self.model.eval()

        if self.args.dataset == 'imdb':
            h = self.model.init_hidden( self.args.batch_size )
        for i, ( input, target ) in enumerate( self.val_dl ):
            if self.args.device == 'gpu':
                input, target = input.cuda(), target.cuda()

            with torch.no_grad():
                if self.args.dataset == 'imdb':
                    h = tuple( [ each.data for each in h ] )
                    # h = [ tuple( [ each.data for each in hi ] ) for hi in h ]
                    output, h = self.model( input, h )
                else:
                    output = self.model( input )
                loss = self.criterion( output, target.float() )

            if self.args.dataset == 'imdb':
                prec1 = cal_acc_binary( output, target )
            else:
                prec1 = cal_acc( output, target )[ 0 ]
            avg_loss.update( loss.item(), input.size( 0 ) )
            avg_acc.update( prec1.item(), input.size( 0 ) )

        self.logger.info('Client: {}\t Epoch: [{}]\tLoss {:.4f}\tPrec@1 {:.3f}'.format(
            self.client_index, epoch, avg_loss.avg, avg_acc.avg
            )
        )

        return avg_acc, avg_loss
