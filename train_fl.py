"""Model decomposition in FL
Project description:
    This script implement model decomposition in Federated Learning (FL) environment.
    Each client trains a subset of models (principal channels).
    The subsets of model parameters are aggregated on the server side.

Note:
"""
import argparse
import os
import sys
import time
import torch
import logging
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.tensorboard import SummaryWriter

from fl.env import Context

sys.path.insert( 0, './' )

parser = argparse.ArgumentParser()
parser.add_argument( '--root-dir', default='./', type=str )
parser.add_argument(
    '--model', default='resnet18', type=str, help='model name'
)
parser.add_argument(
    '--device', default='gpu', type=str, choices=['gpu', 'cpu'], help='untrusted platform'
)
parser.add_argument(
    '--dataset', default='cifar10', type=str, choices=['cifar10', 'femnist', 'imdb']
)

# model decomposition hyperparam
parser.add_argument(
    '--channel-keep', default='1.0', type=float, help='keep rate in orthogonal channels'
)
parser.add_argument(
    '--drop-orthogonal', action='store_true', help='whether to drop orthogonal channels'
)
parser.add_argument(
    '--drop-original', action='store_true', help='whether to drop original channels'
)
parser.add_argument(
    '--random-mask', action='store_true', help='whether mask channels in a random way'
)

# FL hyperparam
parser.add_argument(
    '--n-clients', default=10, type=int, help='number of total clients'
)
parser.add_argument(
    '--prob-factor', default=1.0, type=float, help='factor controls sampling probability'
)
parser.add_argument(
    '--hetero', action='store_true', help='whether in heterogeneous settings'
)
parser.add_argument(
    '--client-capacities', default=[ 0.2, 0.4 ], nargs="+", type=float,
    help='heterogeneous clients capacities'
)
parser.add_argument(
    '--client-ratios', default=[ 0.6, 0.4 ], nargs="+", type=float,
    help='ratios of various heterogeneous clients'
)
parser.add_argument(
    '--active-ratio', default=0.2, type=float, help='ratio of active clients'
)
parser.add_argument(
    '--distribution', default='iid', type=str, choices=['iid', 'noniid'], help='i.i.d or non i.i.d data'
)
parser.add_argument(
    '--alpha', default=1.0, type=float, help='parameter for Dirichlet distribution'
)
parser.add_argument(
    '--local-epoch', default=2, type=int, help='number of local training epoch before aggregating'
)
parser.add_argument(
    '--local-bs', default=32, type=int, help='batch size of local training'
)
parser.add_argument(
    '--aggr-mode', default='FedAVG', type=str, choices=['FedAVG'], help='method of model aggregation'
)
parser.add_argument(
    '--global-bs', default=128, type=int, help='batch size of global validation'
)
parser.add_argument(
    '--total-round', default=400, type=int, help='total global training rounds'
)
parser.add_argument(
    '--warmup-round', default=50, type=int, help='warmup round for fixed dropout'
)
parser.add_argument( '--lr', default=0.1, type=float )
parser.add_argument( '--decay', default='cos', choices=[ 'cos', 'multisteps' ] )
parser.add_argument( '--momentum', default=0.9, type=float )
parser.add_argument( '--wd', default=0.0002, type=float )

# training progress
parser.add_argument( '--check-freq', default=50, type=int )
parser.add_argument( '--save-dir', default='./checkpoints', type=str )
parser.add_argument( '--logdir', default='log/fl/resnet18/orig' )

args = parser.parse_args()

logFormatter = logging.Formatter( "%(asctime)s [%(levelname)-5.5s]  %(message)s", datefmt='%m-%d,%H:%M:%S' )
root_logger = logging.getLogger()
root_logger.setLevel( logging.DEBUG )

if not os.path.exists( args.logdir ):
    os.makedirs( args.logdir )
fileHandler = logging.FileHandler("{0}/{1}.log".format( args.logdir, 'train' ) )
fileHandler.setFormatter( logFormatter )
root_logger.addHandler( fileHandler )

consoleHandler = logging.StreamHandler( sys.stdout )
consoleHandler.setFormatter( logFormatter )
root_logger.addHandler( consoleHandler )

# print all elements in tensor for debugging purpose
torch.set_printoptions( edgeitems=sys.maxsize )


def main():
    root_logger.info( '-' * 80 )
    root_logger.info( 'Federated model decomposition' )
    root_logger.info( '-' * 80 )

    for k, v in args.__dict__.items():
        root_logger.info( '    - {} : {}'.format( k, v ) )

    writer = SummaryWriter( log_dir=args.logdir )

    # -------------------- Init a FL environment -------------------- #
    fl_context = Context( args, logger=root_logger, writer=writer )
    fl_context.init_server()
    fl_context.init_clients()

    # ---------------------- Start FL training ---------------------- #
    fl_context.fl_train( writer )

    writer.close()


if __name__ == '__main__':
    main()
