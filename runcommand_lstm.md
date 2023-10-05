## Running command in federated settings

#### original model

```shell
CUDA_VISIBLE_DEVICES=2 python train_fl.py \
    --lr=0.1 --local-bs=32 \
    --local-epoch=2 --n-clients=100 --active-ratio=0.2 --total-round=500\
    --dataset=imdb --distribution=noniid --alpha=0.1\
    --logdir=log/fl/lstm/noniid/alpha_01/tmp_orig_bs32_r500_ep2_cl100_active0.2
```

```shell
CUDA_VISIBLE_DEVICES=3 python train_fl.py \
    --lr=0.1 --local-bs=32 --wd=0.0001 \
    --drop-original --channel-keep=0.8 \
    --local-epoch=2 --n-clients=100 --active-ratio=0.2 --total-round=500\
    --dataset=imdb --distribution=noniid --alpha=0.1 \
    --logdir=log/fl/lstm/noniid/alpha_01/tmp_orig_keep0.8_r500_ep2_cl100_0.2_wd1
```

#### orthogonal model

- random mask
```shell
CUDA_VISIBLE_DEVICES=6 python train_fl.py \
    --lr=0.1 --local-bs=32 \
    --drop-orthogonal --random-mask --channel-keep=0.4 --prob-factor=2.5 \
    --local-epoch=2 --n-clients=100 --active-ratio=0.2 --total-round=500\
    --dataset=imdb --distribution=noniid --alpha=Inf \
    --logdir=log/fl/lstm/noniid/alpha_Inf/tmp_orth_keep0.4_random_r500_ep2_cl100_0.2_aggrmask
```

```shell
CUDA_VISIBLE_DEVICES=2 python train_fl.py \
    --lr=0.1 --local-bs=32 \
    --drop-orthogonal --random-mask --channel-keep=0.2 --prob-factor=2.5 \
    --local-epoch=2 --n-clients=100 --active-ratio=0.2 --total-round=500\
    --dataset=imdb --distribution=noniid --alpha=0.1 \
    --logdir=log/fl/lstm/noniid/alpha_01/orth_keep0.2_random_r500_ep2_cl100_0.2_aggrmask
```

- deterministic mask
```shell
CUDA_VISIBLE_DEVICES=3 python train_fl.py \
    --lr=0.1 --local-bs=32 \
    --drop-orthogonal --channel-keep=0.8 \
    --local-epoch=2 --n-clients=100 --active-ratio=0.2 --total-round=300\
    --dataset=imdb --distribution=noniid --alpha=Inf \
    --logdir=log/fl/lstm/noniid/alpha_Inf/orth_keep0.8_r300_ep2_cl100_0.2_aggrmask
```

```shell
CUDA_VISIBLE_DEVICES=3 python train_fl.py \
    --lr=0.1 --local-bs=32 \
    --drop-orthogonal --channel-keep=0.8 \
    --local-epoch=2 --n-clients=100 --active-ratio=0.2 --total-round=500\
    --dataset=imdb --distribution=noniid --alpha=0.1 \
    --logdir=log/fl/lstm/noniid/alpha_01/orth_keep0.8_r500_ep2_cl100_0.2_aggrmask
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