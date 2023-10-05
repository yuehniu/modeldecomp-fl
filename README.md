## Intro
This is the official implementation for the paper 
"Overcoming Resource Constraints in Federated Learning: Large Models Can Be Trained with only Weak Clients".

## Dependencies
- torch>=1.10
- numpy>=1.17
- tensorboard>=2.6
- torchvision>0.10

## Run in federated settings

### via bash

#### orthogonal model on homogeneous clients

alpha = Inf, 1, 0.1
```shell
CUDA_VISIBLE_DEVICES=0 python3.10 train_fl.py \
    --wd=0.0005 \
    --drop-orthogonal --random-mask --channel-keep=1.0 --prob-factor=2.5 \
    --local-epoch=2 --n-clients=100 --active-ratio=0.2 --total-round=800 \
    --distribution=noniid --alpha=Inf \
    --logdir=log/fl/resnet18/iid/orth_keep1.0_random_r800_ep2_cl100_0.2
```

#### orthogonal model on heterogeneous clients

```shell
CUDA_VISIBLE_DEVICES=3 python train_fl.py \
    --wd=0.0005 \
    --drop-orthogonal --random-mask --hetero --prob-factor=3 \
    --client-capacities 0.2 0.4 0.6 --client-ratios 0.5 0.3 0.2 \
    --local-epoch=2 --n-clients=100 --active-ratio=0.2 --total-round=1000\
    --distribution=noniid --alpha=0.1 \
    --logdir=log/fl/resnet18/hetero/alpha_01/orth_max0.6_random_r1000_ep2_cl100_0.2_wd5_scaling-no_ps3_fro-s_bn-false
```

## Citation

If you found our works useful, please consider citing the following works:

```
@article{niu2022federated,
  title={Federated Learning of Large Models at the Edge via Principal Sub-Model Training},
  author={Yue Niu, Saurav Prakash, Souvik Kundu, Sunwoo Lee, Salman Avestimehr},
  journal={arXiv preprint arXiv:2208.13141},
  year={2022}
}

@article{niu2023tmlr,
  title={Overcoming Resource Constraints in Federated Learning: Large Models Can Be Trained with only Weak Clients},
  author={Yue Niu, Saurav Prakash, Souvik Kundu, Sunwoo Lee, Salman Avestimehr},
  journal={Transaction on Machine Learning Research (TMLR)},
  year={2023}
}
```