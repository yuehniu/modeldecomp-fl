#!/bin/bash
gpus="1"
model=deit_tiny
round=600
alpha=Inf
droporth=0
droporig=0
randommask=0
keep=0.2
p=4
wd=2e-5

if [ "$droporth" -eq 1 ]; then
  moreopts="--drop-orthogonal --channel-keep=${keep}  --prob-factor=${p}"
  logdir="log/fl/${model}/noniid_${alpha}/orth_keep${keep}_r${round}_ep2_cl100_p${p}_wd${wd}"
  if [ "$randommask" -eq 1 ]; then
    moreopts="${moreopts} --random-mask"
    logdir="${logdir}_rand"
  fi
elif [ "${droporig}" -eq 1 ]; then
  moreopts="--drop-original --channel-keep=${keep}"
  logdir="log/fl/${model}/noniid_${alpha}/orig_keep${keep}_r${round}_ep2_cl100_wd${wd}"
else
  moreopts=""
  logdir="log/fl/${model}/noniid_${alpha}/orig_r${round}_ep2_cl100_wd${wd}_pretrain"
fi
# logdir="log/fl/${model}/runtime/orig_0.2x_r${round}_ep2_cl100_wd${wd}"

if [ -d "$logdir" ]; then
  rm -r $logdir
fi
mkdir -p $logdir

CUDA_VISIBLE_DEVICES=$gpus python train_fl.py \
    --model=$model $moreopts\
    --wd=$wd --device=gpu \
    --local-epoch=2 --n-clients=100 --active-ratio=0.2 --total-round=$round\
    --distribution=noniid --alpha=$alpha \
    --logdir=$logdir