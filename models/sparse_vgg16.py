import tensorflow_probability as tfp
from models.layers import *


class SparseVGG16(object):
    def __init__(self, num_classes, p_global, name='vgg16'):
        super(SparseVGG16, self).__init__()

        units = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512]
        self.num_classes = num_classes
        self.layers = []

        p_init = [0.6, 0.6, 0.2, 0.2, 0.5 * p_global, 0.5 * p_global, 0.5 * p_global,
                  0.5 * p_global, 0.5 * p_global, 0.5 * p_global, 0.5 * p_global,
                  0.5 * p_global, 0.5 * p_global, 0.5 * p_global, 0.5 * p_global, 0.6]

        with tf.name_scope(name):
            in_channels, i = 3, 1
            for v in units:
                if isinstance(v, int):
                    self.layers.append(Conv2DRewiring(in_channels, v, p_init=p_init[i - 1], k_size=3, padding='SAME',
                                                      use_bias=False, name='conv' + str(i)))
                    self.layers.append(BatchNorm(v, name='bn' + str(i)))
                    in_channels, i = v, i + 1
                elif v == "M":
                    self.layers.append(MaxPool(k_size=2, strides=2))
                else:
                    pass
            self.layers.append(Flatten())
            self.layers.append(DenseRewiring(in_channels * 2 * 2, 256, p_init=p_init[13], use_bias=True, name='dense14'))
            self.layers.append(DenseRewiring(256, 256, p_init=p_init[14], use_bias=True, name='dense15'))
            self.layers.append(DenseRewiring(256, num_classes, p_init=p_init[15], use_bias=True, name='dense16'))

        # Useful metrics for connectivity
        self.full_units = [tf.size(lay).numpy() for lay in self.rewiring_param_list()]
        self.dist = tfp.distributions.Categorical(probs=[n / sum(self.full_units) for n in self.full_units])
        self.global_total = tf.cast(sum(self.full_units) * p_global, tf.int32)
        self.p_global = p_global

    def __call__(self, x, train=True):
        for layer in self.layers[:-1]:
            if isinstance(layer, BatchNorm) or isinstance(layer, DenseRewiring):
                x = layer(x, train=train, activation=tf.nn.relu)
            else:
                x = layer(x, train=train, activation=None)
        return self.layers[-1](x, train=train, activation=None)

    def param_list(self, trainable=True):
        params = []
        for layer in self.layers:
            params += layer.params(trainable=trainable)
        return params

    def rewiring_param_list(self):
        params = []
        for layer in self.layers:
            if isinstance(layer, DenseRewiring) or isinstance(layer, Conv2DRewiring):
                params += layer.rewiring_params()
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
