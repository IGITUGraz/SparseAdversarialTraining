import os
import numpy as np
import pickle
import scipy.io as sio
import tensorflow as tf
import tensorflow_addons as tfa
from keras.preprocessing.image import ImageDataGenerator
from models import *


def data_loader(args):
    def data_conversion(d_set):
        images, labels = d_set
        images = images / 255.0
        return tf.cast(images, tf.float32), tf.cast(labels[:, 0], tf.int64)

    pix_range = [0., 1.]    # hard-coded: change if data_conversion function above changes

    if args.data == 'cifar10':
        train_set, test_set = tf.keras.datasets.cifar10.load_data()
        train_tuple, test_tuple = map(data_conversion, [train_set, test_set])
    elif args.data == 'cifar100':
        train_set, test_set = tf.keras.datasets.cifar100.load_data()
        train_tuple, test_tuple = map(data_conversion, [train_set, test_set])
    elif args.data == 'svhn':
        def prep_svhn(dict):
            images, labels = dict['X'], dict['y']
            images = np.moveaxis(images, 3, 0)
            labels[labels % 10 == 0] = 0
            return images, labels
        path = 'datasets/SVHN/'
        train_set, test_set = map(prep_svhn, [sio.loadmat(path + 'train_32x32.mat'), sio.loadmat(path + 'test_32x32.mat')])
        train_tuple, test_tuple = map(data_conversion, [train_set, test_set])
    else:
        raise NotImplementedError

    return train_tuple, test_tuple, pix_range


def data_aux_loader(aux_batch_size):
    data = pickle.load(open("datasets/tinyimages/ti_500K_pseudo_labeled.pickle", "rb"))
    x_train_aux, y_train_aux = data["data"], data["extrapolated_targets"]
    x_train_aux, y_train_aux = tf.cast(x_train_aux / 255.0, tf.float32), tf.cast(y_train_aux, tf.int64)
    aux_iterator = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1,
                                      horizontal_flip=True).flow(x_train_aux, y_train_aux,
                                                                 shuffle=True, batch_size=aux_batch_size)
    return aux_iterator


def model_loader(args):
    if args.model == "vgg16":
        model = SparseVGG16(num_classes=args.n_classes, p_global=args.connectivity) if args.sparse_train \
            else VGG16(num_classes=args.n_classes)
    elif args.model == "resnet18":
        model = SparseResNet18(num_classes=args.n_classes, p_global=args.connectivity) if args.sparse_train \
            else ResNet18(num_classes=args.n_classes)
    elif args.model == "resnet34":
        model = SparseResNet34(num_classes=args.n_classes, p_global=args.connectivity) if args.sparse_train \
            else ResNet34(num_classes=args.n_classes)
    elif args.model == "resnet50":
        model = SparseResNet50(num_classes=args.n_classes, p_global=args.connectivity) if args.sparse_train \
            else ResNet50(num_classes=args.n_classes)
    elif args.model == "resnet101":
        model = SparseResNet101(num_classes=args.n_classes, p_global=args.connectivity) if args.sparse_train \
            else ResNet101(num_classes=args.n_classes)
    elif args.model == "wrn28_2":
        model = SparseWideResNet28_2(num_classes=args.n_classes, p_global=args.connectivity) if args.sparse_train \
            else WideResNet28_2(num_classes=args.n_classes)
    elif args.model == "wrn28_4":
        model = SparseWideResNet28_4(num_classes=args.n_classes, p_global=args.connectivity) if args.sparse_train \
            else WideResNet28_4(num_classes=args.n_classes)
    elif args.model == "wrn28_10":
        model = SparseWideResNet28_10(num_classes=args.n_classes, p_global=args.connectivity) if args.sparse_train \
            else WideResNet28_10(num_classes=args.n_classes)
    elif args.model == "wrn34_10":
        model = SparseWideResNet34_10(num_classes=args.n_classes, p_global=args.connectivity) if args.sparse_train \
            else WideResNet34_10(num_classes=args.n_classes)
    else:
        raise NotImplementedError

    return model


def scheduler_loader(args, iter_per_epoch):
    lr_schedule = [args.l_rate, args.l_rate * 1e-1, args.l_rate * 1e-2]
    wd_schedule = [args.w_decay, args.w_decay * 1e-1, args.w_decay * 1e-2]
    lr_schedule_bounds = [int(iter_per_epoch * args.n_epochs * bound) for bound in [.5, .75]]
    lr_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(lr_schedule_bounds, lr_schedule)
    wd_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(lr_schedule_bounds, wd_schedule)
    optimizer = tfa.optimizers.SGDW(learning_rate=lr_fn, weight_decay=wd_fn, momentum=0.9)
    return lr_fn, wd_fn, optimizer


def get_filename(args):
    if not os.path.exists(os.path.join('results', args.data, args.model)):
        os.makedirs(os.path.join('results', args.data, args.model))
    if args.sparse_train:
        file_name = 'sparse'
        if args.connectivity * 100 >= 1:
            file_name += str(int(args.connectivity * 100))
        else:   # assuming connectivity even less than 1% (e.g., 99.5% sparsity)
            file_name += '0' + str(int(args.connectivity * 1000))
    else:
        file_name = 'full'
    file_name += '_' + args.objective
    return os.path.join(args.data, args.model, file_name)
