import tensorflow_probability as tfp
from models.layers import *


class SparseBasicResidual(object):
    expansion = 1

    def __init__(self, in_units, out_units, p_init, stride, name='block'):
        super(SparseBasicResidual, self).__init__()
        self.layers = []
        self.shortcut = []

        with tf.name_scope(name):
            self.layers.append(Conv2DRewiring(in_units, out_units, p_init=p_init, k_size=3, strides=stride,
                                              padding='SAME', use_bias=False, name='conv1'))
            self.layers.append(BatchNorm(out_units, name='bn1'))
            self.layers.append(Conv2DRewiring(out_units, out_units, p_init=p_init, k_size=3, strides=1, padding='SAME',
                                              use_bias=False, name='conv2'))
            self.layers.append(BatchNorm(out_units, name='bn2'))

            if stride != 1 or in_units != self.expansion * out_units:
                self.shortcut.append(Conv2DRewiring(in_units, self.expansion * out_units, p_init=p_init, k_size=1,
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

    def rewiring_param_list(self):
        params = []
        for layer in self.layers + self.shortcut:
            if isinstance(layer, DenseRewiring) or isinstance(layer, Conv2DRewiring):
                params += layer.rewiring_params()
        return params


class SparseBottleneck(object):
    expansion = 4

    def __init__(self, in_units, out_units, p_init, stride, name='block'):
        super(SparseBottleneck, self).__init__()
        self.layers = []
        self.shortcut = []

        with tf.name_scope(name):
            self.layers.append(Conv2DRewiring(in_units, out_units, p_init=p_init, k_size=1, strides=1,
                                              padding='VALID', use_bias=False, name='conv1'))
            self.layers.append(BatchNorm(out_units, name='bn1'))
            self.layers.append(Conv2DRewiring(out_units, out_units, p_init=p_init, k_size=3, strides=stride,
                                              padding='SAME', use_bias=False, name='conv2'))
            self.layers.append(BatchNorm(out_units, name='bn2'))
            self.layers.append(Conv2DRewiring(out_units, self.expansion * out_units, p_init=p_init, k_size=1, strides=1,
                                              padding='VALID', use_bias=False, name='conv3'))
            self.layers.append(BatchNorm(self.expansion * out_units, name='bn3'))

            if stride != 1 or in_units != self.expansion * out_units:
                self.shortcut.append(Conv2DRewiring(in_units, self.expansion * out_units, p_init=p_init, k_size=1,
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

    def rewiring_param_list(self):
        params = []
        for layer in self.layers + self.shortcut:
            if isinstance(layer, DenseRewiring) or isinstance(layer, Conv2DRewiring):
                params += layer.rewiring_params()
        return params


class SparseResNet(object):
    def __init__(self, block, num_blocks, p_global, num_classes=10, name='resnet'):
        super(SparseResNet, self).__init__()
        self.num_classes = num_classes
        self.layers = []
        self.units = 64

        p_init = [0.6, 0.2, 0.5 * p_global, 0.5 * p_global, 0.5 * p_global, 0.6]

        with tf.name_scope(name):
            self.layers.append(Conv2DRewiring(3, 64, p_init=p_init[0], k_size=3, strides=1, padding='SAME',
                                              use_bias=False, name='conv1'))
            self.layers.append(BatchNorm(64, name='bn1'))
            self._make_block(block, 64, num_blocks[0], p_init=p_init[1], first_stride=1)
            self._make_block(block, 128, num_blocks[1], p_init=p_init[2], first_stride=2)
            self._make_block(block, 256, num_blocks[2], p_init=p_init[3], first_stride=2)
            self._make_block(block, 512, num_blocks[3], p_init=p_init[4], first_stride=2)
            self.layers.append(AvgPool(k_size=4, strides=4))
            self.layers.append(Flatten())
            self.layers.append(DenseRewiring(512 * block.expansion, num_classes, p_init=p_init[5],
                                             use_bias=True, name='dense_out'))

        # Useful metrics for connectivity
        self.full_units = [tf.size(lay).numpy() for lay in self.rewiring_param_list()]
        self.dist = tfp.distributions.Categorical(probs=[n / sum(self.full_units) for n in self.full_units])
        self.global_total = tf.cast(sum(self.full_units) * p_global, tf.int32)
        self.p_global = p_global

    def _make_block(self, block, units, num_blocks, p_init, first_stride):
        strides = [first_stride] + [1] * (num_blocks - 1)
        for stride in strides:
            self.layers.append(block(self.units, units, p_init, stride))
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
            if isinstance(layer, SparseBasicResidual) or isinstance(layer, SparseBottleneck):
                params += layer.param_list(trainable=trainable)
            else:
                params += layer.params(trainable=trainable)
        return params

    def rewiring_param_list(self):
        params = []
        for layer in self.layers:
            if isinstance(layer, SparseBasicResidual) or isinstance(layer, SparseBottleneck):
                params += layer.rewiring_param_list()
            elif isinstance(layer, DenseRewiring) or isinstance(layer, Conv2DRewiring):
                params += layer.rewiring_params()
            else:
                pass
        return params

    def sample_connectivity(self):
        def rewire_params(theta, n_reconnect, epsilon=1e-12):
            is_con = tf.greater(theta.read_value(), 0)
            n_reconnect = tf.maximum(n_reconnect, 0)
            reconnect_candidate_coord = tf.where(tf.logical_not(is_con))

            n_candidates = tf.shape(reconnect_candidate_coord)[0]
            reconnect_sample_id = tf.random.shuffle(tf.range(n_candidates))[:n_reconnect]
            reconnect_sample_coord = tf.gather(reconnect_candidate_coord, reconnect_sample_id)
            reconnect_vals = tf.fill(dims=[n_reconnect], value=epsilon)
            theta.scatter_nd_update(reconnect_sample_coord, reconnect_vals)

        rewire_n_units = self.get_rewire_n_units()
        [rewire_params(th, nz) for th, nz in zip(self.rewiring_param_list(), rewire_n_units)]

    def get_rewire_n_units(self):
        n_connected = [nb_connected(th.read_value()) for th in self.rewiring_param_list()]
        total_connected = tf.reduce_sum(n_connected)
        nb_reconnect = tf.maximum(0, self.global_total - total_connected)
        sample_split = self.dist.sample(nb_reconnect)
        n_reconnect = [tf.reduce_sum(tf.cast(tf.equal(sample_split, i), tf.int32)) for i in range(len(n_connected))]
        return n_reconnect

    def connectivity_stats(self):
        n_connected = [int(nb_connected(th.read_value())) for th in self.rewiring_param_list()]
        layer_conn = [float(conn / size) for conn, size in zip(n_connected, self.full_units)]
        global_conn = tf.reduce_sum(n_connected) / tf.reduce_sum(self.full_units) * 100
        return global_conn, layer_conn, np.array(n_connected)


def SparseResNet18(**kwargs):
    return SparseResNet(SparseBasicResidual, [2, 2, 2, 2], name='resnet18', **kwargs)


def SparseResNet34(**kwargs):
    return SparseResNet(SparseBasicResidual, [3, 4, 6, 3], name='resnet34', **kwargs)


def SparseResNet50(**kwargs):
    return SparseResNet(SparseBottleneck, [3, 4, 6, 3], name='resnet50', **kwargs)
