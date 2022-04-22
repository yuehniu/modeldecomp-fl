# Introduction
In this work ...

# Environment
- Torch>=1.10


## Running command in centralized settings

### Orthogonal channel dropout

```shell
# keep 20% channel for every convolutional layer
python train.py --drop-orthogonal --channel-keep=0.2 \
    --momentum=0.6 --epochs=200 \
    --logdir=log/resnet18/**
```


### Regular channel dropout
```shell
# keep 20% channel for every convolutional layer
python train.py --drop-regular --channel-keep=0.2 \
    --logdir=log/resnet18/**
```

## Running command in federated settings

### i.i.d datasets

#### original model

```shell
CUDA_VISIBLE_DEVICES=1 python train_fl.py \
    --local-epoch=2 --n-clients=100 \
    --logdir=log/fl/resnet18/iid/orig_r400_ep2_cl100_active0.2
```

#### orthogonal model
```shell
CUDA_VISIBLE_DEVICES=2 python train_fl.py \
    --drop-orthogonal --random-mask \
    --local-epoch=2 --n-clients=100 \
    --logdir=log/fl/resnet18/iid/orth_r400_ep2_cl100_active0.2
```

### Non i.i.d datasets

#### original model

```shell
CUDA_VISIBLE_DEVICES=0 python train_fl.py \
    --local-epoch=2 --n-clients=100 --active-ratio=0.1 \
    --distribution=noniid --alpha=1.0 \
    --logdir=log/fl/resnet18/noniid/alpha1/orig_r400_ep2_cl100_active0.1_alphaInf
```

#### orthogonal model

- random mask
```shell
CUDA_VISIBLE_DEVICES=0 python train_fl.py \
    --wd=0.0005 \
    --drop-orthogonal --random-mask --channel-keep=1.0 \
    --local-epoch=2 --n-clients=100 --active-ratio=0.2\
    --distribution=noniid --alpha=Inf \
    --logdir=log/tmp/orth_keep1.0_random_r400_ep2_cl100_0.2_wd5e-4_s
```

- deterministic mask
```shell
CUDA_VISIBLE_DEVICES=1 python train_fl.py \
    --drop-orthogonal --channel-keep=0.6 \
    --lr=0.1 --local-epoch=2 --n-clients=100 --active-ratio=0.2\
    --distribution=noniid --alpha=Inf \
    --logdir=log/fl/resnet18/noniid/alphaInf/orth_keep0.6+_r400_ep2_cl100_0.2_aggrUV
```