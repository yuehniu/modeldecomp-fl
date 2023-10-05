## Running command in federated settings

#### original model

```shell
CUDA_VISIBLE_DEVICES=0 python train_fl.py \
    --lr=0.01 --local-bs=32 \
    --local-epoch=2 --n-clients=100 --active-ratio=0.2 --total-round=100\
    --dataset=femnist --distribution=noniid\
    --logdir=log/fl/cnn/noniid/orig_bs32_r100_ep2_cl100_active0.2
```

```shell
CUDA_VISIBLE_DEVICES=3 python train_fl.py \
    --lr=0.01 --local-bs=32 --wd=0.0005 \
    --drop-original --channel-keep=0.2 \
    --local-epoch=2 --n-clients=100 --active-ratio=0.2 --total-round=100\
    --dataset=femnist --distribution=noniid \
    --logdir=log/fl/cnn/noniid/orig_keep0.2_random_r100_ep2_cl100_0.2_wd5
```

#### orthogonal model

- random mask
```shell
CUDA_VISIBLE_DEVICES=1 python train_fl.py \
    --lr=0.01 --local-bs=32 --wd=0.0002 \
    --drop-orthogonal --random-mask --channel-keep=0.8 --prob-factor=2.5 \
    --local-epoch=2 --n-clients=100 --active-ratio=0.2 --total-round=100\
    --dataset=femnist --distribution=noniid \
    --logdir=log/fl/cnn2/noniid/orth_keep0.8_random_r100_ep2_cl100_0.2_wd2_ps2.5
```

- deterministic mask
```shell
CUDA_VISIBLE_DEVICES=7 python train_fl.py \
    --lr=0.01 --local-bs=32 --wd=0.0002 \
    --drop-orthogonal --channel-keep=0.2 \
    --local-epoch=2 --n-clients=100 --active-ratio=0.2 --total-round=100\
    --dataset=femnist --distribution=noniid \
    --logdir=log/fl/cnn/noniid/orth_keep0.2_r100_ep2_cl100_0.2_wd2_scaling-no_fro-s
```


#### orthogonal model on heterogeneous clients

- random mask

```shell
CUDA_VISIBLE_DEVICES=1 python train_fl.py \
    --lr=0.01 --local-bs=32 --wd=0.0005 \
    --drop-orthogonal --random-mask --hetero --prob-factor=2.5 \
    --local-epoch=2 --n-clients=100 --active-ratio=0.2 --total-round=100\
    --dataset=femnist --distribution=noniid \
    --logdir=log/fl/cnn/hetero/noniid/orth_max0.6_random_r100_ep2_cl100_0.2_wd5_scaling-no_ps2.5_fro-s
```

- deterministic mask
```shell
CUDA_VISIBLE_DEVICES=3 python train_fl.py \
    --lr=0.01 --local-bs=32 --wd=0.0005 \
    --drop-orthogonal --hetero \
    --local-epoch=2 --n-clients=100 --active-ratio=0.2 --total-round=100\
    --dataset=femnist --distribution=noniid \
    --logdir=log/fl/cnn/hetero/noniid/orth_max0.6_r100_ep2_cl100_0.2_wd5_scaling-no_fro-s
```

- original dropout
```shell
CUDA_VISIBLE_DEVICES=1 python train_fl.py \
    --lr=0.01 --local-bs=32 --wd=0.0005 \
    --drop-original --hetero \
    --local-epoch=2 --n-clients=100 --active-ratio=0.2 --total-round=100\
    --dataset=femnist --distribution=noniid \
    --logdir=log/fl/cnn/hetero/noniid/orig_max0.6_r100_ep2_cl100_0.2_wd5_scaling-no_fro-s
```