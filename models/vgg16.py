from models.layers import *


class VGG16(object):
    def __init__(self, num_classes, name='vgg16'):
        super(VGG16, self).__init__()

        units = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512]
        self.num_classes = num_classes
        self.layers = []

        with tf.name_scope(name):
            in_channels, kernel_id = 3, 1
            for v in units:
                if isinstance(v, int):
                    self.layers.append(Conv2D(in_channels, v, k_size=3, padding='SAME', use_bias=False,
                                              name='conv' + str(kernel_id)))
                    self.layers.append(BatchNorm(v, name='bn' + str(kernel_id)))
                    in_channels, kernel_id = v, kernel_id + 1
                elif v == "M":
                    self.layers.append(MaxPool(k_size=2, strides=2))
                else:
                    pass
            self.layers.append(Flatten())
            self.layers.append(Dense(in_channels * 2 * 2, 256, use_bias=True, name='dense14'))
            self.layers.append(Dense(256, 256, use_bias=True, name='dense15'))
            self.layers.append(Dense(256, num_classes, use_bias=True, name='dense16'))

    def __call__(self, x, train=True):
        for layer in self.layers[:-1]:
            if isinstance(layer, BatchNorm) or isinstance(layer, Dense):
                x = layer(x, train=train, activation=tf.nn.relu)
            else:
                x = layer(x, train=train, activation=None)
        return self.layers[-1](x, train=train, activation=None)

    def param_list(self, trainable=True):
        params = []
        for layer in self.layers:
            params += layer.params(trainable=trainable)
        return params
