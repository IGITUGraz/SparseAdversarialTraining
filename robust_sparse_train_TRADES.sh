#!/bin/bash
dt=$(date '+%d/%m/%Y %H:%M:%S');
echo $dt

dataset="cifar100"
arch="vgg16"
classes=100
wd=5e-4

# Train robust and sparse models with TRADES
python -u run_connectivity_sampling.py --data $dataset --model $arch --n_classes $classes -s -pc 0.01 -wd $wd --objective "trades"
python -u run_connectivity_sampling.py --data $dataset --model $arch --n_classes $classes -s -pc 0.1 -wd $wd --objective "trades"
python -u run_connectivity_sampling.py --data $dataset --model $arch --n_classes $classes -wd $wd --objective "trades"

# Evaluate white box adversarial robustness with TRADES
python -u run_foolbox_eval.py --data $dataset --model $arch --n_classes $classes -s -pc 0.01 --objective "trades"
python -u run_foolbox_eval.py --data $dataset --model $arch --n_classes $classes -s -pc 0.1 --objective "trades"
python -u run_foolbox_eval.py --data $dataset --model $arch --n_classes $classes --objective "trades"
