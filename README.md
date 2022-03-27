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
    --local-epoch=5 \
    --logdir=log/fl/resnet18/orig_r400_ep5_cl10_active2
```