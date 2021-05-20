from models.layers import *


class BasicResidual(object):
    expansion = 1

    def __init__(self, in_units, out_units, stride, name='block'):
        super(BasicResidual, self).__init__()
        self.layers = []
        self.shortcut = []

        with tf.name_scope(name):
            self.layers.append(Conv2D(in_units, out_units, k_size=3, strides=stride, padding='SAME',
                                      use_bias=False, name='conv1'))
            self.layers.append(BatchNorm(out_units, name='bn1'))
            self.layers.append(Conv2D(out_units, out_units, k_size=3, strides=1, padding='SAME',
                                      use_bias=False, name='conv2'))
            self.layers.append(BatchNorm(out_units, name='bn2'))

            if stride != 1 or in_units != self.expansion * out_units:
                self.shortcut.append(Conv2D(in_units, self.expansion * out_units, k_size=1,
                                            strides=stride, padding='VALID', use_bias=False, name='conv3'))
                self.shortcut.append(BatchNorm(self.expansion * out_units, name='bn3'))

    def __call__(self, x, train=True):
        out = self.layers[0](x, train=train, activation=None)
        out = self.layers[1](out, train=train, activation=tf.nn.relu)
        out = self.layers[2](out, train=train, activation=None)
        out = self.layers[3](out, train=train, activation=None)
        x_sc = x
        for ly in range(len(self.shortcut)):
            x_sc = self.shortcut[ly](x_sc, train=train, activation=None)
        return tf.nn.relu(out + x_sc)

    def param_list(self, trainable=True):
        params = []
        for layer in self.layers + self.shortcut:
            params += layer.params(trainable=trainable)
        return params


class Bottleneck(object):
    expansion = 4

    def __init__(self, in_units, out_units, stride, name='block'):
        super(Bottleneck, self).__init__()
        self.layers = []
        self.shortcut = []

        with tf.name_scope(name):
            self.layers.append(Conv2D(in_units, out_units, k_size=1, strides=1, padding='VALID',
                                      use_bias=False, name='conv1'))
            self.layers.append(BatchNorm(out_units, name='bn1'))
            self.layers.append(Conv2D(out_units, out_units, k_size=3, strides=stride, padding='SAME',
                                      use_bias=False, name='conv2'))
            self.layers.append(BatchNorm(out_units, name='bn2'))
            self.layers.append(Conv2D(out_units, self.expansion * out_units, k_size=1, strides=1,
                                      padding='VALID', use_bias=False, name='conv3'))
            self.layers.append(BatchNorm(self.expansion * out_units, name='bn3'))

            if stride != 1 or in_units != self.expansion * out_units:
                self.shortcut.append(Conv2D(in_units, self.expansion * out_units, k_size=1,
                                            strides=stride, padding='VALID', use_bias=False, name='conv4'))
                self.shortcut.append(BatchNorm(self.expansion * out_units, name='bn4'))

    def __call__(self, x, train=True):
        out = self.layers[0](x, train=train, activation=None)
        out = self.layers[1](out, train=train, activation=tf.nn.relu)
        out = self.layers[2](out, train=train, activation=None)
        out = self.layers[3](out, train=train, activation=tf.nn.relu)
        out = self.layers[4](out, train=train, activation=None)
        out = self.layers[5](out, train=train, activation=None)
        x_sc = x
        for ly in range(len(self.shortcut)):
            x_sc = self.shortcut[ly](x_sc, train=train, activation=None)
        return tf.nn.relu(out + x_sc)

    def param_list(self, trainable=True):
        params = []
        for layer in self.layers + self.shortcut:
            params += layer.params(trainable=trainable)
        return params


class ResNet(object):
    def __init__(self, block, num_blocks, num_classes=10, name='resnet'):
        super(ResNet, self).__init__()
        self.num_classes = num_classes
        self.layers = []
        self.units = 64

        with tf.name_scope(name):
            self.layers.append(Conv2D(3, 64, k_size=3, strides=1, padding='SAME', use_bias=False, name='conv1'))
            self.layers.append(BatchNorm(64, name='bn1'))
            self._make_block(block, 64, num_blocks[0], first_stride=1)
            self._make_block(block, 128, num_blocks[1], first_stride=2)
            self._make_block(block, 256, num_blocks[2], first_stride=2)
            self._make_block(block, 512, num_blocks[3], first_stride=2)
            self.layers.append(AvgPool(k_size=4, strides=4))
            self.layers.append(Flatten())
            self.layers.append(Dense(512 * block.expansion, num_classes, use_bias=True, name='dense_out'))

    def _make_block(self, block, units, num_blocks, first_stride):
        strides = [first_stride] + [1] * (num_blocks - 1)
        for stride in strides:
            self.layers.append(block(self.units, units, stride))
            self.units = units * block.expansion

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
            if isinstance(layer, BasicResidual) or isinstance(layer, Bottleneck):
                params += layer.param_list(trainable=trainable)
            else:
                params += layer.params(trainable=trainable)
        return params


def ResNet18(**kwargs):
    return ResNet(BasicResidual, [2, 2, 2, 2], name='resnet18', **kwargs)


def ResNet34(**kwargs):
    return ResNet(BasicResidual, [3, 4, 6, 3], name='resnet34', **kwargs)


def ResNet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], name='resnet50', **kwargs)


def ResNet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3],  name='resnet101', **kwargs)
