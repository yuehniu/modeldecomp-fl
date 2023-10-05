## Running command in federated settings

### i.i.d datasets

#### original model

```shell
CUDA_VISIBLE_DEVICES=1 python train_fl.py \
   --model=resnet8 \
    --local-epoch=2 --n-clients=100 -active-ratio=0.2 --total-round=600 \
    --logdir=log/fl/resnet18/iid/orig_r400_ep2_cl100_active0.2
```

#### orthogonal model
```shell
CUDA_VISIBLE_DEVICES=2 python train_fl.py \
    --drop-orthogonal --random-mask \
    --local-epoch=2 --n-clients=100 \
    --logdir=log/fl/resnet18/iid/temp-orth_r400_ep2_cl100_active0.2
```

### Non i.i.d datasets

#### original model

```shell
CUDA_VISIBLE_DEVICES=4 python train_fl.py \
    --model=resnet8 \
    --local-epoch=2 --n-clients=100 --active-ratio=0.2 --total-round=600\
    --distribution=noniid --alpha=Inf \
    --logdir=log/fl/resnet8/noniid/alpha_inf/orig_r600_ep2_cl100_active0.2
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
    --model=resnet18 --wd=0.0002 --local-bs=32 \
    --drop-orthogonal --random-mask --channel-keep=0.8 --prob-factor=2.5 \
    --local-epoch=3 --n-clients=100 --active-ratio=0.2 --total-round=2400\
    --distribution=noniid --alpha=Inf \
    --logdir=log/fl/resnet18/iid/orth_keep0.8_random_r2400_ep3_cl100_0.2_wd2_bs32_p4_fixU2_noS
```

alpha = 1
```shell
CUDA_VISIBLE_DEVICES=0 python train_fl.py \
    --model=resnet18 --wd=0.0002 \
    --drop-orthogonal --random-mask --channel-keep=0.2 --prob-factor=4 \
    --local-epoch=3 --n-clients=100 --active-ratio=0.2 --total-round=2400\
    --distribution=noniid --alpha=1 \
    --logdir=log/fl/resnet18/noniid_1/orth_keep0.2+_random_r2400_ep3_cl100_0.2_wd2_ps4
```

alpha = 0.1
```shell
CUDA_VISIBLE_DEVICES=1 python train_fl.py \
    --model=resnet18 --wd=0.0002 \
    --drop-orthogonal --random-mask --channel-keep=0.2 --prob-factor=4 \
    --local-epoch=3 --n-clients=100 --active-ratio=0.2 --total-round=2400\
    --distribution=noniid --alpha=0.1 \
    --logdir=log/fl/resnet18/noniid_01/orth_keep0.2+_random_r2400_ep3_cl100_0.2_wd2_ps4
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
    --model=resnet18 --wd=0.0002 \
    --drop-orthogonal --random-mask --hetero --prob-factor=4 \
    --local-epoch=3 --n-clients=100 --active-ratio=0.2 --total-round=2400 \
    --distribution=noniid --alpha=1 \
    --logdir=log/fl/resnet18/hetero/iid_1/orth_max0.4+_random_r2400_ep3_cl100_0.2_wd2_ps4_2
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