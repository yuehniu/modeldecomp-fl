"""
Project decription:
Note:
"""
import argparse
import os
import sys
import time
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, './')
from utils.nn import build_network, convert_to_orth_model, update_orth_channel, add_regular_dropout
from data.dataset import *

parser = argparse.ArgumentParser()
parser.add_argument( 
    '--model', default='resnet18', type=str,
    help='model name' 
)
parser.add_argument( 
    '--device', default='gpu', type=str, choices=['gpu', 'cpu'], 
    help='untrusted platform' 
)
parser.add_argument( 
    '--dataset', default='cifar10', type=str, choices=['cifar10', 'cifar100', 'imagenet']
)
parser.add_argument(
    '--channel-keep', default='0.8', type=float,
    help='keep rate in orthogonal channels'
)
parser.add_argument(
    '--drop-orthogonal', action='store_true',
    help='whether to drop orthogonal channels'
)
parser.add_argument(
    '--drop-regular', action='store_true',
    help='whether to drop regular convolutional channels'
)
parser.add_argument( '--root-dir', default='./', type=str )
parser.add_argument( '--batch-size', default=128, type=int )
parser.add_argument( '--lr', default=0.1, type=float )
parser.add_argument( '--decay', default='cos', choices=[ 'cos', 'multisteps' ] )
parser.add_argument( '--momentum', default=0.9, type=float )
parser.add_argument( '--wd', default=0.0002, type=float )
parser.add_argument( '--epochs', default=200, type=int )
parser.add_argument( '--workers', default=4, type=int )
parser.add_argument( '--check-freq', default=20, type=int )
parser.add_argument( '--save-dir', default='./checkpoints', type=str )
parser.add_argument(
    '--pre-train', default=None, type=str,
    help='pretrained model'
)
parser.add_argument( '--logdir', default='log/resnet18/orig' )
args = parser.parse_args()


def main():
    # build model
    model = None
    train_set, test_set = None, None
    if args.dataset == 'cifar10':
        model = build_network( args.model, len_feature=512, num_classes=10 )  # ResNet:64 VGG:512
        # sz_img = 32
        train_set = dataset_CIFAR10_train
        test_set = dataset_CIFAR10_test
    elif args.dataset == 'imagenet':
        len_feature = 512
        if 'vgg' in args.model:
            len_feature = 4608
        elif 'resnet' in args.model:
            len_feature = 512
        model = build_network( args.model, len_feature=len_feature, num_classes=1000 )  # ResNet: 256 VGG:4608
        # sz_img = 224
        train_set = dataset_IMAGENET_train
        test_set = dataset_IMAGENET_test

    assert model is not None, 'A model has not been defined!'
    if args.pre_train is not None:
        model.load_state_dict( torch.load( args.pre_train )[ 'state_dict' ] )

    if args.drop_orthogonal:
        model = convert_to_orth_model( model, args.channel_keep )
    elif args.drop_regular:
        model = add_regular_dropout( model, drop=1-args.channel_keep )
    # update_orth_channel( model, dropout=args.channel_dropout )
    print( model )

    if args.device == 'gpu':
        model.cuda()
    else:
        model.cpu()

    # construct dataset
    assert train_set is not None, 'a training dataset has not been defined!'
    assert test_set is not None, 'a test dataset has not been defined!'
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.workers, pin_memory=True 
    )
    val_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size, shuffle=False, drop_last=True,
        num_workers=args.workers, pin_memory=True 
    )

    # define loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    if args.device == 'gpu':
        criterion.cuda()
    else:
        criterion.cpu()
    optimizer = torch.optim.SGD(
        model.parameters(), args.lr,
        momentum=args.momentum, weight_decay=args.wd
    )

    # lr scheduler
    if args.decay == 'cos':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.epochs, eta_min=0.001
        )
    elif args.decay == 'multisteps':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[ 50, 100, 150 ], gamma=0.1
        )
    else:
        raise ValueError( 'Unexpected lr scheduler!!!' )

    # start training
    writer = SummaryWriter( log_dir=args.logdir )
    best_acc = 0.0
    for epoch in range(0, args.epochs):
        prec1_train, loss_train = train(
            model, train_loader, criterion, optimizer, epoch
        )

        prec1_val, loss_val = validate(
            model, val_loader, criterion, epoch
        )

        lr_scheduler.step( epoch=epoch )

        # update orthogonal channel after every epoch
        if args.drop_orthogonal:
            print( 'Update orthogonal channel with keep rate: ', args.channel_keep )
            with torch.no_grad():
                update_orth_channel( model, optimizer, keep=args.channel_keep )

        # store training record
        writer.add_scalar( 'train/loss', loss_train.avg, epoch )
        writer.add_scalar( 'test/acc', prec1_val.avg, epoch )

        # update best accuracy
        if prec1_val.avg > best_acc:
            best_acc = prec1_val.avg

        # save checkpoint
        """
        if epoch % args.check_freq == 0:
            state = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'prec1': prec1_val,
            }
            file_name = os.path.join(args.save_dir, 'checkpoint_{}.tar'.format(epoch))
            torch.save(state, file_name)
        """
    writer.close()
    print( 'Best accuracy: ', best_acc )


def train( model, train_loader, criterion, optimizer, epoch ):
    batch_time, data_time = AverageMeter(), AverageMeter()
    avg_loss, avg_acc = AverageMeter(), AverageMeter()

    model.train()

    end = time.time()
    # TODO: resolve how to update momentum
    for i, ( input, target ) in enumerate( train_loader ):
        # record data loading time
        data_time.update( time.time() - end )

        # input = torch.ones( ( args.batch_size, 3, 32, 32 ), dtype=torch.float32 )

        if args.device == 'gpu':
            input = input.cuda()
            target = target.cuda()

        optimizer.zero_grad()

        # forward and backward
        output = model( input )
        loss = criterion( output, target )
        loss.backward()

        # update parameter
        optimizer.step()

        # report acc and loss
        prec1 = cal_acc( output, target )[ 0 ]

        avg_acc.update( prec1.item(), input.size(0) )
        avg_loss.update( loss.item(), input.size(0) )

        # record elapsed time
        batch_time.update( time.time() - end )
        end = time.time()

        if i % args.check_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {avg_loss.avg:.4f}\t'
                  'Prec@1 {avg_acc.avg:.3f}'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, avg_loss=avg_loss, avg_acc=avg_acc))
    return avg_acc, avg_loss


def validate( model, val_loader, criterion, epoch ):
    avg_loss = AverageMeter()
    avg_acc = AverageMeter()

    model.eval()

    for i, ( input, target ) in enumerate( val_loader ):
        if args.device == 'gpu':
            input = input.cuda()
            target = target.cuda()

        # forward
        with torch.no_grad():
            output = model( input )
            loss = criterion( output, target )

        prec1 = cal_acc( output, target )[ 0 ]
        avg_loss.update( loss.item(), input.size(0) )
        avg_acc.update( prec1.item(), input.size(0) )

    print('Epoch: [{}]\tLoss {:.4f}\tPrec@1 {:.3f}'.format( epoch, avg_loss.avg, avg_acc.avg ) )

    return avg_acc, avg_loss


def cal_acc( output, target, topk=(1,) ):
    """
    Calculate model accuracy
    :param output:
    :param target:
    :param topk:
    :return: topk accuracy
    """
    maxk = max( topk )
    batch_size = target.size( 0 )

    _, pred = output.topk( maxk, 1, True, True )
    pred = pred.t()
    correct = pred.eq( target.view(1, -1).expand_as(pred) )

    acc = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        acc.append(correct_k.mul_(100.0 / batch_size))
    return acc


class AverageMeter( object ):
    """Computes and stores the average and current value"""
    def __init__( self ):
        self.reset()

    def reset( self ):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update( self, val, n=1 ):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()
