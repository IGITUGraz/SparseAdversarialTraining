import tensorflow as tf
import numpy as np

he_normal = tf.initializers.he_normal()
zeros = tf.initializers.zeros()
ones = tf.initializers.ones()


def get_actual_weights(theta, sign):
    return tf.where(condition=tf.greater(theta, 0), x=tf.multiply(sign, theta), y=tf.zeros(sign.shape, dtype=tf.float32))


def nb_connected(theta):
    return tf.reduce_sum(tf.cast(tf.greater(theta, 0), tf.int32))


def initialize_conn_mask(fan_in, fan_out, p_init):
    non_zeros = int(p_init * fan_in * fan_out)
    conn_mask = np.zeros([fan_in * fan_out], dtype=bool)
    conn_mask[np.random.choice(fan_in * fan_out, non_zeros, replace=False)] = True
    conn_mask = np.reshape(conn_mask, [fan_in, fan_out])
    return conn_mask


class Dense(object):
    def __init__(self, n_in, n_out, name='dense', use_bias=False):
        self.use_bias = use_bias
        with tf.name_scope(name):
            self.W = tf.Variable(he_normal(shape=[n_in, n_out], dtype=tf.float32), name="W")
            if use_bias:
                self.b = tf.Variable(zeros(shape=[n_out], dtype=tf.float32), name="b")

    def __call__(self, x, train=True, activation=None):
        x = tf.matmul(x, self.W)
        x += self.b if self.use_bias else x
        x = activation(x) if activation is not None else x
        return x

    def params(self, trainable):
        if self.use_bias:
            return [self.W, self.b]
        else:
            return [self.W]


class Conv2D(object):
    def __init__(self, n_in, n_out, k_size, strides=1, padding='SAME', name='conv', use_bias=False):
        self.use_bias = use_bias
        self.strides = strides
        self.padding = padding
        with tf.name_scope(name):
            self.W = tf.Variable(he_normal(shape=[k_size, k_size, n_in, n_out], dtype=tf.float32), name="W")
            if use_bias:
                self.b = tf.Variable(zeros(shape=[n_out], dtype=tf.float32), name="b")

    def __call__(self, x, train=True, activation=None):
        x = tf.nn.conv2d(x, self.W, strides=self.strides, padding=self.padding)
        x = tf.nn.bias_add(x, self.b) if self.use_bias else x
        x = activation(x) if activation is not None else x
        return x

    def params(self, trainable):
        if self.use_bias:
            return [self.W, self.b]
        else:
            return [self.W]


class DenseRewiring(object):
    def __init__(self, n_in, n_out, p_init, name='dense', use_bias=False):
        self.use_bias = use_bias

        # Initialize weight re-parameterization
        conn_mask = initialize_conn_mask(n_in, n_out, p_init)
        theta_init = tf.abs(he_normal(shape=[n_in, n_out], dtype=tf.float32))
        sign_init = tf.sign(tf.random.normal(shape=[n_in, n_out], dtype=tf.float32))

        with tf.name_scope(name):
            self.theta = tf.Variable(theta_init * conn_mask, name="T")
            self.sign = tf.Variable(sign_init, name='S', trainable=False)
            if use_bias:
                self.b = tf.Variable(zeros(shape=[n_out], dtype=tf.float32), name="b")

    def __call__(self, x, train=True, activation=None):
        W = get_actual_weights(self.theta, self.sign)
        x = tf.matmul(x, W)
        x += self.b if self.use_bias else x
        x = activation(x) if activation is not None else x
        return x

    def params(self, trainable=True):
        params = [self.theta]
        params += [self.sign] if not trainable else []
        params += [self.b] if self.use_bias else []
        return params

    def rewiring_params(self):
        return [self.theta]


class Conv2DRewiring(object):
    def __init__(self, n_in, n_out, p_init, k_size, strides=1, padding='SAME', name='conv', use_bias=False):
        self.use_bias = use_bias
        self.strides = strides
        self.padding = padding

        # Initialize weight re-parameterization
        fan_in, fan_out = int(k_size * k_size * n_in), n_out
        conn_mask = initialize_conn_mask(fan_in, fan_out, p_init)
        theta_init = tf.abs(he_normal(shape=[fan_in, fan_out], dtype=tf.float32))
        sign_init = tf.sign(tf.random.normal(shape=[k_size, k_size, n_in, n_out], dtype=tf.float32))

        with tf.name_scope(name):
            self.theta = tf.Variable(tf.reshape(theta_init * conn_mask, shape=[k_size, k_size, n_in, n_out]), name="T")
            self.sign = tf.Variable(sign_init, name='S', trainable=False)
            if use_bias:
                self.b = tf.Variable(zeros(shape=[n_out], dtype=tf.float32), name="b")

    def __call__(self, x, train=True, activation=None):
        W = get_actual_weights(self.theta, self.sign)
        x = tf.nn.conv2d(x, W, strides=self.strides, padding=self.padding)
        x = tf.nn.bias_add(x, self.b) if self.use_bias else x
        x = activation(x) if activation is not None else x
        return x

    def params(self, trainable=True):
        params = [self.theta]
        params += [self.sign] if not trainable else []
        params += [self.b] if self.use_bias else []
        return params

    def rewiring_params(self):
        return [self.theta]


class BatchNorm(object):
    def __init__(self, n_in, momentum=0.9, epsilon=1e-5, name='bn'):
        self.momentum = momentum
        self.epsilon = epsilon
        with tf.name_scope(name):
            self.beta = tf.Variable(zeros(shape=[n_in], dtype=tf.float32), name="beta")
            self.gamma = tf.Variable(ones(shape=[n_in], dtype=tf.float32), name="gamma")
            self.running_mean = tf.Variable(zeros(shape=[n_in], dtype=tf.float32), trainable=False, name="running_mean")
            self.running_var = tf.Variable(ones(shape=[n_in], dtype=tf.float32), trainable=False, name="running_var")

    def __call__(self, x, train=True, activation=None):
        if train:
            if len(x.shape) == 4:
                batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2])
            else:
                batch_mean, batch_var = tf.nn.moments(x, [0])
            self.running_mean.assign_sub((1 - self.momentum) * (self.running_mean - batch_mean))
            self.running_var.assign_sub((1 - self.momentum) * (self.running_var - batch_var))
            x = tf.nn.batch_normalization(x, batch_mean, batch_var, self.beta, self.gamma, self.epsilon)
        else:
            x = tf.nn.batch_normalization(x, self.running_mean, self.running_var, self.beta, self.gamma, self.epsilon)
        x = activation(x) if activation is not None else x
        return x

    def params(self, trainable=True):
        if trainable:
            return [self.gamma, self.beta]
        else:
            return [self.gamma, self.beta, self.running_mean, self.running_var]


class MaxPool(object):
    def __init__(self, k_size=2, strides=2, padding='VALID'):
        self.k_size = k_size
        self.strides = strides
        self.padding = padding

    def __call__(self, x, train=True, activation=None):
        return tf.nn.max_pool(x, ksize=[1, self.k_size, self.k_size, 1],
                              strides=[1, self.strides, self.strides, 1],
                              padding=self.padding)

    def params(self, trainable):
        return []


class AvgPool(object):
    def __init__(self, k_size=2, strides=2, padding='VALID'):
        self.k_size = k_size
        self.strides = strides
        self.padding = padding

    def __call__(self, x, train=True, activation=None):
        return tf.nn.avg_pool(x, ksize=[1, self.k_size, self.k_size, 1],
                              strides=[1, self.strides, self.strides, 1],
                              padding=self.padding)

    def params(self, trainable):
        return []


class Flatten(object):
    def __init__(self):
        pass

    def __call__(self, x, train=True, activation=None):
        return tf.reshape(x, [-1, np.prod(x.shape.as_list()[1:])])

    def params(self, trainable):
        return []
