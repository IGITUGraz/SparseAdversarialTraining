#!/bin/bash
dt=$(date '+%d/%m/%Y %H:%M:%S');
echo $dt

dataset="cifar10"
arch="resnet50"
classes=10
wd=1e-3

# Train robust and sparse models with Standard AT
python -u run_connectivity_sampling.py --data $dataset --model $arch --n_classes $classes -s -pc 0.01 -wd $wd --objective "at"
python -u run_connectivity_sampling.py --data $dataset --model $arch --n_classes $classes -s -pc 0.1 -wd $wd --objective "at"
python -u run_connectivity_sampling.py --data $dataset --model $arch --n_classes $classes -wd $wd --objective "at"

# Evaluate white box adversarial robustness with Standard AT
python -u run_foolbox_eval.py --data $dataset --model $arch --n_classes $classes -s -pc 0.01 --objective "at"
python -u run_foolbox_eval.py --data $dataset --model $arch --n_classes $classes -s -pc 0.1 --objective "at"
python -u run_foolbox_eval.py --data $dataset --model $arch --n_classes $classes --objective "at"
