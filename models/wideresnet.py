from models.layers import *


class BasicBlock(object):
    def __init__(self, in_units, out_units, stride, name='block'):
        super(BasicBlock, self).__init__()
        self.layers = []
        self.shortcut = []
        self.equalInOut = (in_units == out_units)

        with tf.name_scope(name):
            self.layers.append(BatchNorm(in_units, name='bn1'))
            self.layers.append(Conv2D(in_units, out_units, k_size=3, strides=stride, padding='SAME',
                                      use_bias=False, name='conv1'))
            self.layers.append(BatchNorm(out_units, name='bn2'))
            self.layers.append(Conv2D(out_units, out_units, k_size=3, strides=1, padding='SAME',
                                      use_bias=False, name='conv2'))
            if not self.equalInOut:
                self.shortcut.append(Conv2D(in_units, out_units, k_size=1, strides=stride, padding='VALID',
                                            use_bias=False, name='conv3'))

    def __call__(self, x, train=True):
        if not self.equalInOut:
            x = self.layers[0](x, train=train, activation=tf.nn.relu)
            out = self.layers[1](x, train=train, activation=None)
        else:
            out = self.layers[0](x, train=train, activation=tf.nn.relu)
            out = self.layers[1](out, train=train, activation=None)
        out = self.layers[2](out, train=train, activation=tf.nn.relu)   # assume no dropout after this step
        out = self.layers[3](out, train=train, activation=None)
        return out + (x if self.equalInOut else self.shortcut[0](x, train=train, activation=None))

    def param_list(self, trainable=True):
        params = []
        for layer in self.layers + self.shortcut:
            params += layer.params(trainable=trainable)
        return params


class NetworkBlock(object):
    def __init__(self, nb_layers, in_units, out_units, block, stride, name='block'):
        super(NetworkBlock, self).__init__()
        self.layers = []

        with tf.name_scope(name):
            self.layers.append(block(in_units, out_units, stride))
            for i in range(1, int(nb_layers)):
                self.layers.append(block(out_units, out_units, 1))

    def __call__(self, x, train=True):
        for layer in self.layers:
            x = layer(x, train=train)
        return x

    def param_list(self, trainable=True):
        params = []
        for layer in self.layers:
            params += layer.param_list(trainable=trainable)
        return params


class WideResNet(object):
    def __init__(self, depth, widen_factor, num_classes=10, name='wideresnet'):
        super(WideResNet, self).__init__()
        self.num_classes = num_classes
        self.layers = []
        units = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        n = (depth - 4) / 6

        with tf.name_scope(name):
            self.layers.append(Conv2D(3, units[0], k_size=3, strides=1, padding='SAME', use_bias=False, name='conv1'))
            self.layers.append(NetworkBlock(n, units[0], units[1], BasicBlock, stride=1))
            self.layers.append(NetworkBlock(n, units[1], units[2], BasicBlock, stride=2))
            self.layers.append(NetworkBlock(n, units[2], units[3], BasicBlock, stride=2))
            self.layers.append(BatchNorm(units[3], name='bn1'))
            self.layers.append(AvgPool(k_size=8, strides=8))
            self.layers.append(Flatten())
            self.layers.append(Dense(units[3], num_classes, use_bias=True, name='dense_out'))

    def __call__(self, x, train=True):
        for layer in self.layers:
            if isinstance(layer, BatchNorm):
                x = layer(x, train=train, activation=tf.nn.relu)
            else:
                x = layer(x, train=train)
        return x

    def param_list(self, trainable=True):
        params = []
        for layer in self.layers:
            if isinstance(layer, BasicBlock) or isinstance(layer, NetworkBlock):
                params += layer.param_list(trainable=trainable)
            else:
                params += layer.params(trainable=trainable)
        return params


def WideResNet28_2(**kwargs):
    return WideResNet(depth=28, widen_factor=2, name='wideresnet-28-2', **kwargs)


def WideResNet28_4(**kwargs):
    return WideResNet(depth=28, widen_factor=4, name='wideresnet-28-4', **kwargs)


def WideResNet28_10(**kwargs):
    return WideResNet(depth=28, widen_factor=10, name='wideresnet-28-10', **kwargs)


def WideResNet34_10(**kwargs):
    return WideResNet(depth=34, widen_factor=10, name='wideresnet-34-10', **kwargs)
