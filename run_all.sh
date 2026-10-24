#!/bin/bash
# Stops each run once test PSNR reaches the paper value (-0.05).
# Train + test DepthLux on LOL-v1, LOL-v2-real, LOL-v2-synthetic (~30k steps each, batch 4).
PY=${PY:-python}; D=${DATA:-../data}; mkdir -p logs
export CUDA_CACHE_MAXSIZE=4294967296
run() { # name dataset root epochs paper_psnr
  $PY train.py --data_root $3 --dataset $2 --epochs $4 --eval_every 5 --target_psnr $5 --out checkpoints > logs/$1.train.log 2>&1
  $PY evaluate.py --data_root $3 --dataset $2 --weights checkpoints/$2_best.weights.h5 > logs/$1.test.log 2>&1
}
run lolv1      lolv1      $D/lolv1/lol_dataset      250 26.06
run lolv2_real lolv2_real $D/hf_lolv2-real          175 26.16
run lolv2_syn  lolv2_syn  $D/hf_lolv2-synthetic     135 28.69
