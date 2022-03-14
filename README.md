# Training Adversarially Robust Sparse Networks via Bayesian Connectivity Sampling [ICML 2021]

This is the code repository of the following [paper](http://proceedings.mlr.press/v139/ozdenizci21a/ozdenizci21a.pdf) for end-to-end robust adversarial training of neural networks with sparse connectivity.
 
"Training Adversarially Robust Sparse Networks via Bayesian Connectivity Sampling"\
<em>Ozan Özdenizci, Robert Legenstein</em>\
International Conference on Machine Learning (ICML), 2021.

The repository supports sparse training of models with the robust training objectives explored in the paper, as well as saved model weights of the adversarially trained sparse networks that are presented.

## Setup

You will need [TensorFlow 2](https://www.tensorflow.org/install) to run this code. You can simply start by executing:
```bash
pip install -r requirements.txt
```
to install all dependencies and use the repository.

## Usage

You can use `run_connectivity_sampling.py` to adversarially train sparse networks from scratch. Brief description of possible arguments are:

- `--data`: "cifar10", "cifar100", "svhn"
- `--model`: "vgg16", "resnet18", "resnet34", "resnet50", "wrn28_2", "wrn28_4", "wrn28_10", "wrn34_10"
- `--objective`: "at" (Standard AT), "mat" (Mixed-batch AT), trades", "mart", "rst" (intended for CIFAR-10)
- `--sparse_train`: enable end-to-end sparse training
- `--connectivity`: sparse connectivity ratio (used when `--sparse_train` is enabled)

Remarks:
* For the `--data "svhn"` option, you will need to create the directory `datasets/SVHN/` and place the [SVHN](http://ufldl.stanford.edu/housenumbers/) dataset's [train](http://ufldl.stanford.edu/housenumbers/train_32x32.mat) and [test](http://ufldl.stanford.edu/housenumbers/test_32x32.mat) `.mat` files there.
* We consider usage of robust self-training (RST) `--objective "rst"` based on the TRADES loss. To be able to use RST for CIFAR-10 as described in [this repository](https://github.com/yaircarmon/semisup-adv), you need to place the [pseudo-labeled TinyImages](https://drive.google.com/open?id=1LTw3Sb5QoiCCN-6Y5PEKkq9C9W60w-Hi) file at `datasets/tinyimages/ti_500K_pseudo_labeled.pickle`.

### End-to-end robust training for sparse networks

The following sample scripts can be used to adversarially train sparse networks from scratch, and also perform white box robustness evaluations using PGD attacks via [Foolbox](https://github.com/bethgelab/foolbox).

- `robust_sparse_train_standardAT.sh`: Standard adversarial training for a sparse ResNet-50 on CIFAR-10.
- `robust_sparse_train_TRADES.sh` Robust training with TRADES for a sparse VGG-16 on CIFAR-100.

## Saved model weights

We share our adversarially trained sparse models at 90% and 99% sparsity for CIFAR-10, CIFAR-100 and SVHN datasets that are presented in the paper. 
Different evaluations may naturally result in slight differences in the numbers presented in the paper.

### Sparse networks with TRADES robust training objective

* CIFAR-10  (TRADES with RST): 
[VGG16 - 90% Sparsity](https://igi-web.tugraz.at/download/OzdenizciLegensteinICML2021/cifar10_vgg16_sparse10_rst.zip) | 
[VGG16 - 99% Sparsity](https://igi-web.tugraz.at/download/OzdenizciLegensteinICML2021/cifar10_vgg16_sparse1_rst.zip)
* CIFAR-100 (TRADES): 
[ResNet-18 - 90% Sparsity](https://igi-web.tugraz.at/download/OzdenizciLegensteinICML2021/cifar100_resnet18_sparse10_trades.zip) | 
[ResNet-18 - 99% Sparsity](https://igi-web.tugraz.at/download/OzdenizciLegensteinICML2021/cifar100_resnet18_sparse1_trades.zip)
* SVHN   (TRADES): 
[WideResNet-28-4 - 90% Sparsity](https://igi-web.tugraz.at/download/OzdenizciLegensteinICML2021/svhn_wrn28_4_sparse10_trades.zip) | 
[WideResNet-28-4 - 99% Sparsity](https://igi-web.tugraz.at/download/OzdenizciLegensteinICML2021/svhn_wrn28_4_sparse1_trades.zip)

### Sparse networks with Standard AT for CIFAR-10

These sparse models trained with standard AT on CIFAR-10 (without additional pseudo-labeled images) that correspond to our models presented in Figure 1 and Table 4 of the paper.

* VGG16      : 
[90% Sparsity](https://igi-web.tugraz.at/download/OzdenizciLegensteinICML2021/cifar10_vgg16_sparse10_at.zip) | 
[99% Sparsity](https://igi-web.tugraz.at/download/OzdenizciLegensteinICML2021/cifar10_vgg16_sparse1_at.zip) | 
[99.5% Sparsity](https://igi-web.tugraz.at/download/OzdenizciLegensteinICML2021/cifar10_vgg16_sparse05_at.zip)
* ResNet-18  : 
[99% Sparsity](https://igi-web.tugraz.at/download/OzdenizciLegensteinICML2021/cifar10_resnet18_sparse1_at.zip)
* ResNet-34  : 
[99% Sparsity](https://igi-web.tugraz.at/download/OzdenizciLegensteinICML2021/cifar10_resnet34_sparse1_at.zip)
* ResNet-50  : 
[99% Sparsity](https://igi-web.tugraz.at/download/OzdenizciLegensteinICML2021/cifar10_resnet50_sparse1_at.zip)

#### An example on how to evaluate saved model weights

Originally we store the learned model weights in pickle dictionaries, however to enable benchmark evaluations on [Foolbox](https://github.com/bethgelab/foolbox) and [AutoAttack](https://github.com/fra31/auto-attack) we convert and load these saved dictionary of weights into equivalent Keras models for compatibility. 

Consider the last pickle file above that corresponds to the ResNet-50 model weights at 99% sparsity trained via Standard AT on CIFAR-10. 
Place this file such that the following directory can be accessed: `results/cifar10/resnet50/sparse1_at_best_weights.pickle`.
You can simply use `run_foolbox_eval.py` to load these network weights into Keras models and evaluate robustness against PGD<sup>50</sup> attacks as follows:
```bash
python run_foolbox_eval.py --data "cifar10" --n_classes 10 --model "resnet50" --objective "at" --sparse_train --connectivity 0.01 --pgd_iters 50 --pgd_restarts 10
```

## Reference
If you use this code or models in your research and find it helpful, please cite the following paper:
```
@inproceedings{ozdenizci2021icml,
  title={Training adversarially robust sparse networks via Bayesian connectivity sampling},
  author={Ozan \"{O}zdenizci and Robert Legenstein},
  booktitle={International Conference on Machine Learning},
  pages={8314--8324},
  year={2021},
  organization={PMLR}
}
```

## Acknowledgments

Authors of this work are affiliated with Graz University of Technology, Institute of Theoretical Computer Science, 
and Silicon Austria Labs, TU Graz - SAL Dependable Embedded Systems Lab, Graz, Austria. This work has been supported by the "University SAL Labs" initiative of Silicon Austria Labs (SAL) and its Austrian partner universities for applied fundamental research for electronic based systems. 
This work is also partially supported by the Austrian Science Fund (FWF) within the ERA-NET CHIST-ERA programme (project SMALL, project number I 4670-N).

Parts of this code repository is based on the following works by the machine learning community.

* https://github.com/guillaumeBellec/deep_rewiring
* https://github.com/inspire-group/hydra
* https://github.com/yaodongyu/TRADES
* https://github.com/YisenWang/MART
* https://github.com/yaircarmon/semisup-adv
