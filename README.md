# Introduction
In this work ...


## Running

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