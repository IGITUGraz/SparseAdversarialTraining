import tensorflow_probability as tfp
from models.layers import *


class SparseBasicBlock(object):
    def __init__(self, in_units, out_units, p_init, stride, name='block'):
        super(SparseBasicBlock, self).__init__()
        self.layers = []
        self.shortcut = []
        self.equalInOut = (in_units == out_units)

        with tf.name_scope(name):
            self.layers.append(BatchNorm(in_units, name='bn1'))
            self.layers.append(Conv2DRewiring(in_units, out_units, p_init=p_init, k_size=3, strides=stride,
                                              padding='SAME', use_bias=False, name='conv1'))
            self.layers.append(BatchNorm(out_units, name='bn2'))
            self.layers.append(Conv2DRewiring(out_units, out_units, p_init=p_init, k_size=3, strides=1, padding='SAME',
                                              use_bias=False, name='conv2'))
            if not self.equalInOut:
                self.shortcut.append(Conv2DRewiring(in_units, out_units, p_init=p_init, k_size=1, strides=stride,
                                                    padding='VALID', use_bias=False, name='conv3'))

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

    def rewiring_param_list(self):
        params = []
        for layer in self.layers + self.shortcut:
            if isinstance(layer, DenseRewiring) or isinstance(layer, Conv2DRewiring):
                params += layer.rewiring_params()
        return params


class SparseNetworkBlock(object):
    def __init__(self, nb_layers, in_units, out_units, block, p_init, stride, name='block'):
        super(SparseNetworkBlock, self).__init__()
        self.layers = []

        with tf.name_scope(name):
            self.layers.append(block(in_units, out_units, p_init, stride))
            for i in range(1, int(nb_layers)):
                self.layers.append(block(out_units, out_units, p_init, 1))

    def __call__(self, x, train=True):
        for layer in self.layers:
            x = layer(x, train=train)
        return x

    def param_list(self, trainable=True):
        params = []
        for layer in self.layers:
            params += layer.param_list(trainable=trainable)
        return params

    def rewiring_param_list(self):
        params = []
        for layer in self.layers:
            params += layer.rewiring_param_list()
        return params


class SparseWideResNet(object):
    def __init__(self, depth, widen_factor, p_global, num_classes=10, name='wideresnet'):
        super(SparseWideResNet, self).__init__()
        self.num_classes = num_classes
        self.layers = []
        units = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        n = (depth - 4) / 6

        p_init = [0.6, 0.1, 0.5 * p_global, 0.5 * p_global, 0.6]

        with tf.name_scope(name):
            self.layers.append(Conv2DRewiring(3, units[0], p_init=p_init[0], k_size=3, strides=1, padding='SAME',
                                              use_bias=False, name='conv1'))
            self.layers.append(SparseNetworkBlock(n, units[0], units[1], SparseBasicBlock, p_init=p_init[1], stride=1))
            self.layers.append(SparseNetworkBlock(n, units[1], units[2], SparseBasicBlock, p_init=p_init[2], stride=2))
            self.layers.append(SparseNetworkBlock(n, units[2], units[3], SparseBasicBlock, p_init=p_init[3], stride=2))
            self.layers.append(BatchNorm(units[3], name='bn1'))
            self.layers.append(AvgPool(k_size=8, strides=8))
            self.layers.append(Flatten())
            self.layers.append(DenseRewiring(units[3], num_classes, p_init=p_init[4], use_bias=True, name='dense_out'))

        # Useful metrics for connectivity
        self.full_units = [tf.size(lay).numpy() for lay in self.rewiring_param_list()]
        self.dist = tfp.distributions.Categorical(probs=[n / sum(self.full_units) for n in self.full_units])
        self.global_total = tf.cast(sum(self.full_units) * p_global, tf.int32)
        self.p_global = p_global

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
            if isinstance(layer, SparseBasicBlock) or isinstance(layer, SparseNetworkBlock):
                params += layer.param_list(trainable=trainable)
            else:
                params += layer.params(trainable=trainable)
        return params

    def rewiring_param_list(self):
        params = []
        for layer in self.layers:
            if isinstance(layer, SparseBasicBlock) or isinstance(layer, SparseNetworkBlock):
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


def SparseWideResNet28_2(**kwargs):
    return SparseWideResNet(depth=28, widen_factor=2, name='wideresnet-28-2', **kwargs)


def SparseWideResNet28_4(**kwargs):
    return SparseWideResNet(depth=28, widen_factor=4, name='wideresnet-28-4', **kwargs)


def SparseWideResNet28_10(**kwargs):
    return SparseWideResNet(depth=28, widen_factor=10, name='wideresnet-28-10', **kwargs)


def SparseWideResNet34_10(**kwargs):
    return SparseWideResNet(depth=34, widen_factor=10, name='wideresnet-34-10', **kwargs)
