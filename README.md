# Introduction
In this work ...


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

### original model

```shell
CUDA_VISIBLE_DEVICES=1 python train_fl.py \
    --local-epoch=2 --n-clients=100 \
    --logdir=log/fl/resnet18/orig_r400_ep2_cl100_active0.2
```

### orthogonal model
```shell
CUDA_VISIBLE_DEVICES=2 python train_fl.py \
    --drop-orthogonal --random-mask \
    --local-epoch=2 --n-clients=100 \
    --logdir=log/fl/resnet18/orth_r400_ep2_cl100_active0.2
```