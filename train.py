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

sys.path.insert(0, './')
from utils.nn import build_network, convert_to_orth_model
from data.dataset import *

parser = argparse.ArgumentParser()
parser.add_argument( 
    '--model', default='vgg16', type=str, 
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
    '--channel-dropout', default='1.0', type=float,
    help='dropout rate in orthonal channels'
)
parser.add_argument(
    '--drop-orthogonal', action='store_true',
    help='whether to drop orthonal channels'
)
parser.add_argument( '--root-dir', default='./', type=str )
parser.add_argument( '--batch-size', default=256, type=int )
parser.add_argument( '--lr', default=0.1, type=float )
parser.add_argument( '--momentum', default=0.9, type=float )
parser.add_argument( '--wd', default=0.0002, type=float )
parser.add_argument( '--epochs', default=150, type=int )
parser.add_argument( '--workers', default=4, type=int )
parser.add_argument( '--check-freq', default=20, type=int )
parser.add_argument( '--save-dir', default='./checkpoints', type=str )

def main():
    global args

    args = parser.parse_args()

    # build model
    if ( args.dataset == 'cifar10' ):
        model = build_network( args.model, len_feature=512, num_classes=10 ) #ResNet:64 VGG:512
        sz_img = 32
        train_set = dataset_CIFAR10_train
        test_set = dataset_CIFAR10_test
    elif ( args.dataset == 'imagenet' ):
        if ( 'vgg' in args.model ):
            len_feature = 4608
        elif ( 'resnet' in args.model ):
            len_feature = 512
        model = build_network( args.model, len_feature=len_feature, num_classes=1000 ) #ResNet: 256 VGG:4608
        sz_img = 224
        train_set = dataset_IMAGENET_train
        test_set = dataset_IMAGENET_test
        
    # convert model if using orthonal channel dropout
    if args.drop_orthogonal:
        model = convert_to_orth_model( model, args.channel_dropout )

    if args.device == 'gpu':
        model.cuda()
    else:
        model.cpu()

    # construct dataset
    normalize = transforms.Normalize( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] )
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.workers, pin_memory=True 
    )
    val_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size, shuffle=False, drop_last = True,
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
        momentum=args.momentum,
        weight_decay=args.wd
    )

    # lr scheduler
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR( optimizer, milestones=[ 50, 100, 150 ], gamma=0.1 )

    # start training
    for epoch in range(0, args.epochs):
        prec1_train, loss_train = train( model, train_loader, criterion, optimizer, epoch )

        prec1_val, loss_val = validate( model, val_loader, criterion, sgxdnn, epoch )

        lr_scheduler.step()

        # save checkpoint
        #if epoch % args.check_freq == 0:
        #    state = {
        #        'epoch': epoch + 1,
        #        'state_dict': model.state_dict(),
        #        'prec1': prec1_val,
        #    }
        #    file_name = os.path.join(args.save_dir, 'checkpoint_{}.tar'.format(epoch))
        #    torch.save(state, file_name)

def train( model, train_loader, criterion, optimizer, epoch ):
    batch_time, data_time = AverageMeter(), AverageMeter()
    avg_loss, avg_acc = AverageMeter(), AverageMeter()

    model.train()

    end = time.time()
    for i, ( input, target ) in enumerate( train_loader ):
        # record data loading time
        data_time.update( time.time() - end )

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
                      epoch, i, len(train_loader), batch_time = batch_time,
                      data_time = data_time, avg_loss = avg_loss, avg_acc = avg_acc))
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
