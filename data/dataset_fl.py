import torch
import torchvision
import numpy as np
from data.dataset import dataset_CIFAR10_train, dataset_CIFAR10_test
from data.common import create_lda_partitions

from typing import (
    Callable, Dict, Generic, Iterable, Iterator,
    List, Optional, Sequence, Tuple, TypeVar,
)

T_co = TypeVar('T_co', covariant=True)


class Subset( torchvision.datasets.VisionDataset ):
    r"""
    Subset of a dataset at specified indices.

    Args:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    def __init__( self, dataset, indices ) -> None:
        super( Subset, self ).__init__( '' )
        self.dataset = dataset
        self.indices = list( indices )

    def __getitem__( self, idx ):
        if isinstance( idx, list ):
            return self.dataset[ [ self.indices[ i ] for i in idx ] ]
        return self.dataset[ self.indices[ idx ] ]

    def __len__(self):
        return len( self.indices )


def create_dataset_fl( args ):
    global_train_dl, global_val_dl = None, None
    local_train_dl, local_val_dl = None, None
    if args.dataset == 'cifar10':
        global_train_set = dataset_CIFAR10_train
        global_val_set = dataset_CIFAR10_test

        # global datasets
        global_train_dl = torch.utils.data.DataLoader(
            global_train_set,
            batch_size=args.global_bs, shuffle=True, drop_last=True,
            num_workers=4, pin_memory=True
        )
        global_val_dl = torch.utils.data.DataLoader(
            global_val_set,
            batch_size=args.global_bs, shuffle=False, drop_last=True,
            num_workers=4, pin_memory=True
        )

        # local datasets
        local_train_indices, local_val_indices = create_distribution( args, global_train_set, global_val_set )
        local_train_dl = [ None for _ in range( args.n_clients ) ]
        local_val_dl = [ None for _ in range( args.n_clients ) ]
        for i in range( args.n_clients ):
            local_train_set = Subset( global_train_set, local_train_indices[ i ] )
            local_val_set = Subset( global_val_set, local_val_indices[ i ] )
            local_train_dl[ i ] = torch.utils.data.DataLoader(
                local_train_set, batch_size=args.local_bs, shuffle=True, drop_last=True
            )
            local_val_dl[ i ] = torch.utils.data.DataLoader(
                local_val_set, batch_size=args.local_bs, shuffle=False, drop_last=True
            )
    else:
        raise NotImplementedError

    return global_train_dl, global_val_dl, local_train_dl, local_val_dl


def create_distribution( args, global_train_set, global_val_set ):
    n_global_train, n_global_val = len( global_train_set ), len( global_val_set )

    if args.distribution == 'iid':
        local_train_indices, local_val_indices = generate_iid( args, n_global_train, n_global_val )
    elif args.distribution == 'noniid':
        local_train_indices, local_val_indices = generate_noniid( args, global_train_set, global_val_set )
    else:
        raise AttributeError

    return local_train_indices, local_val_indices


def generate_iid( args, n_global_train, n_global_val ):
    local_train_indices = [ None for _ in range( args.n_clients ) ]
    local_val_indices = [ None for _ in range( args.n_clients ) ]
    n_local_train, n_local_val = n_global_train // args.n_clients, n_global_val // args.n_clients
    global_train_indices = [ i for i in range( n_global_train ) ]
    global_val_indices = [ i for i in range( n_global_val ) ]

    for i in range( args.n_clients ):
        local_train_indices[ i ] = set(
            np.random.choice( global_train_indices, n_local_train, replace=False )
        )
        global_train_indices = list( set( global_train_indices ) - local_train_indices[ i ] )
        local_val_indices[ i ] = set(
            np.random.choice( global_val_indices, n_local_val, replace=False )
        )
        global_val_indices = list( set( global_val_indices ) - local_val_indices[ i ] )

        # global_train_indices = list( set( global_train_indices ) - local_train_indices[ i ] )
        # global_val_indices = list( set( global_val_indices ) - local_val_indices[ i ] )

    return local_train_indices, local_val_indices


def generate_noniid( args, global_train_set, global_val_set ):
    local_train_indices, dirichlet_dist = create_lda_partitions(
        dataset=np.array( global_train_set.targets ),
        dirichlet_dist=None, num_partitions=args.n_clients,
        concentration=args.alpha, accept_imbalanced=False
    )
    local_train_indices = [ index[ 0 ].tolist() for index in local_train_indices ]

    local_val_indices, _ = create_lda_partitions(
        dataset=np.array( global_val_set.target ),
        dirichlet_dist=dirichlet_dist, num_partitions=args.n_clients,
        concentration=args.alpha, accept_imbalanced=False
    )
    local_val_indices = [ index[ 0 ].tolist() for index in local_val_indices ]

    return local_train_indices, local_val_indices
