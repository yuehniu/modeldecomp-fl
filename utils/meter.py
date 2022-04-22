import torch

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


def cal_entropy( s, n ):
    s = s.view( n, )
    s_normalized = torch.nn.functional.normalize( s, p=1, dim=0 )
    s_sum = torch.sum( s_normalized ** 2 )

    return -torch.log2( s_sum ), torch.max( s )
