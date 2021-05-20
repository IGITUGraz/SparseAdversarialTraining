import tensorflow as tf


def kl_div(l_natural, l_adv):
    return tf.nn.softmax(l_natural) * (tf.nn.log_softmax(l_natural) - tf.nn.log_softmax(l_adv))


def gather_idx(inp, idx):
    locs = tf.stack([tf.expand_dims(tf.range(idx.shape[0], dtype=tf.int64), 1), tf.expand_dims(idx, 1)], axis=-1)
    return tf.gather_nd(inp, tf.squeeze(locs))


def mart_loss(model, x, y, x_adv, beta):
    logits = model(x, train=True)
    logits_adv = model(x_adv, train=True)
    adv_probs = tf.nn.softmax(logits_adv)
    nat_probs = tf.nn.softmax(logits)

    tmp1 = tf.cast(tf.argsort(adv_probs, axis=1)[:, -2:], tf.int64)
    new_y = tf.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1])
    nll_loss = tf.reduce_mean(-tf.math.log(gather_idx(1.0001 - adv_probs, new_y) + 1e-12))
    loss_adv = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y, logits_adv)) + nll_loss
    true_probs = gather_idx(nat_probs, y)
    loss_robust = tf.reduce_sum(tf.reduce_sum(kl_div(logits, logits_adv), axis=1) * (1.0000001 - true_probs)) / x.shape[0]
    train_loss = loss_adv + beta * loss_robust
    train_acc = tf.reduce_mean(tf.cast(tf.equal(y, tf.argmax(logits, axis=1)), dtype=tf.float32))

    return train_acc, train_loss


def mart_adv(model, x, y, eps, eps_iter, nb_iter, clip_range):
    ce_loss = tf.nn.sparse_softmax_cross_entropy_with_logits
    x_adv = (x + 0.001 * tf.random.normal(x.shape, mean=0.0, stddev=1.0))
    for _ in range(nb_iter):
        with tf.GradientTape() as g:
            g.watch(x_adv)
            loss = ce_loss(labels=y, logits=model(x_adv, train=False))
        grad = g.gradient(loss, x_adv)
        x_adv = x_adv + tf.multiply(eps_iter, tf.stop_gradient(tf.sign(grad)))
        x_adv = x + tf.clip_by_value(x_adv - x, -eps, eps)
        x_adv = tf.clip_by_value(x_adv, clip_range[0], clip_range[1])
    x_adv = tf.stop_gradient(tf.clip_by_value(x_adv, clip_range[0], clip_range[1]))

    return x_adv
