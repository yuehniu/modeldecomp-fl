# Introduction
In this work ...

# Environment
- Torch>=1.10

## Running command in federated settings

### via bash
sh run.sh

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
    --local-epoch=2 --n-clients=100 --active-ratio=0.2 --total-round=600\
    --distribution=noniid --alpha=0.1 \
    --logdir=log/fl/resnet18/noniid/alpha_01/tmp_orig_r600_ep2_cl100_active0.2
```

```shell
CUDA_VISIBLE_DEVICES=0 python train_fl.py \
    --lr=0.01 --local-bs=32 \
    --local-epoch=2 --n-clients=100 --active-ratio=0.2 --total-round=100\
    --dataset=femnist --distribution=noniid\
    --logdir=log/fl/cnn/noniid/orig_bs32_r100_ep2_cl100_active0.2
```

alpha = Inf
```shell
CUDA_VISIBLE_DEVICES=0 python train_fl.py \
    --wd=0.0005 \
    --drop-original --channel-keep=0.4 \
    --local-epoch=2 --n-clients=100 --active-ratio=0.2 --total-round=600\
    --distribution=noniid --alpha=Inf \
    --logdir=log/fl/resnet18/noniid/alpha_Inf/orig_keep0.4_random_r600_ep2_cl100_0.2_wd5_bn-false
```

alpha = 1.0
```shell
CUDA_VISIBLE_DEVICES=1 python train_fl.py \
    --wd=0.0005 \
    --drop-original --channel-keep=0.2 \
    --local-epoch=2 --n-clients=100 --active-ratio=0.4 --total-round=600\
    --distribution=noniid --alpha=1 \
    --logdir=log/fl/resnet18/noniid/alpha_1/orig_keep0.4_random_r600_ep2_cl100_0.2_wd5_bn-false
```

alpha = 0.1
```shell
CUDA_VISIBLE_DEVICES=2 python train_fl.py \
    --wd=0.0005 \
    --drop-original --channel-keep=0.4 \
    --local-epoch=2 --n-clients=100 --active-ratio=0.2 --total-round=600\
    --distribution=noniid --alpha=0.1 \
    --logdir=log/fl/resnet18/noniid/alpha_01/orig_keep0.4_random_r600_ep2_cl100_0.2_wd5_bn-false
```

#### orthogonal model

- random mask

alpha = Inf
```shell
CUDA_VISIBLE_DEVICES=0 python3.10 train_fl.py \
    --wd=0.0005 \
    --drop-orthogonal --random-mask --channel-keep=1.0 --prob-factor=2.5 \
    --local-epoch=2 --n-clients=100 --active-ratio=0.2 --total-round=800 \
    --distribution=noniid --alpha=Inf \
    --logdir=log/fl/resnet18/iid/orth_keep1.0_random_r800_ep2_cl100_0.2
```

alpha = 1
```shell
CUDA_VISIBLE_DEVICES=0 python train_fl.py \
    --wd=0.0005 \
    --drop-orthogonal --random-mask --channel-keep=0.8 --prob-factor=2.5 \
    --local-epoch=10 --n-clients=100 --active-ratio=0.2 --total-round=300\
    --distribution=noniid --alpha=1 \
    --logdir=log/fl/resnet18/local-epoch/alpha_1/orth_keep0.8_random_r300_ep10_cl100_0.2
```

alpha = 0.1
```shell
CUDA_VISIBLE_DEVICES=7 python train_fl.py \
    --wd=0.0005 \
    --drop-orthogonal --random-mask --channel-keep=1.0 --prob-factor=3 \
    --local-epoch=2 --n-clients=100 --active-ratio=0.3 --total-round=600\
    --distribution=noniid --alpha=0.1 \
    --logdir=log/fl/resnet18/active-clients/alpha_01/orth_keep1.0_random_r600_ep2_cl100_0.3_ps3
```

- deterministic mask
alpha=Inf
```shell
CUDA_VISIBLE_DEVICES=0 python train_fl.py \
    --wd=0.0005 \
    --drop-orthogonal --channel-keep=0.2 \
    --local-epoch=2 --n-clients=100 --active-ratio=0.2 --total-round=600\
    --distribution=noniid --alpha=Inf \
    --logdir=log/fl/resnet18/noniid/alphaInf/tmp_orth_keep0.2_r800_ep2_cl100_0.2_wd5_scaling-no_fro-s_bn-false
```

alpha = 1
```shell
CUDA_VISIBLE_DEVICES=1 python train_fl.py \
    --wd=0.0005 \
    --drop-orthogonal --channel-keep=0.2 \
    --local-epoch=2 --n-clients=100 --active-ratio=0.2 --total-round=600\
    --distribution=noniid --alpha=1 \
    --logdir=log/fl/resnet18/noniid/alpha_1/orth_keep0.2_r800_ep2_cl100_0.2_wd5_scaling-no_fro-s_bn-false
```

alpha = 0.1
```shell
CUDA_VISIBLE_DEVICES=2 python train_fl.py \
    --wd=0.0005 \
    --drop-orthogonal --channel-keep=0.2 \
    --local-epoch=2 --n-clients=100 --active-ratio=0.2 --total-round=600\
    --distribution=noniid --alpha=0.1 \
    --logdir=log/fl/resnet18/noniid/alpha_01/orth_keep0.2_r600_ep2_cl100_0.2_wd5_scaling-no_fro-s_bn-false
```

#### orthogonal model on heterogeneous clients

- random mask

```shell
CUDA_VISIBLE_DEVICES=3 python train_fl.py \
    --wd=0.0005 \
    --drop-orthogonal --random-mask --hetero --prob-factor=3 \
    --client-capacities 0.2 0.4 0.6 --client-ratios 0.5 0.3 0.2 \
    --local-epoch=2 --n-clients=100 --active-ratio=0.2 --total-round=1000\
    --distribution=noniid --alpha=0.1 \
    --logdir=log/fl/resnet18/hetero/alpha_01/orth_max0.6_random_r1000_ep2_cl100_0.2_wd5_scaling-no_ps3_fro-s_bn-false
```

- deterministic mask
```shell
CUDA_VISIBLE_DEVICES=5 python train_fl.py \
    --wd=0.0005 \
    --drop-orthogonal --hetero \
    --local-epoch=2 --n-clients=100 --active-ratio=0.2 --total-round=600\
    --distribution=noniid --alpha=0.1 \
    --logdir=log/fl/resnet18/hetero/alpha_01/orth_max0.4_r600_ep2_cl100_0.2_wd5_scaling-no_fro-s_bn-false
```

- original dropout
```shell
CUDA_VISIBLE_DEVICES=5 python train_fl.py \
    --wd=0.0005 \
    --drop-original --hetero \
    --local-epoch=2 --n-clients=100 --active-ratio=0.2 --total-round=600\
    --distribution=noniid --alpha=0.1 \
    --logdir=log/fl/resnet18/hetero/alpha_01/orig_max0.4_r600_ep2_cl100_0.2_wd5_scaling-no_fro-s_bn-false
```