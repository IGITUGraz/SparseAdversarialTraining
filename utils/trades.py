import tensorflow as tf


def criterion_kl(l_natural, l_adv):
    return tf.reduce_sum(tf.nn.softmax(l_natural) * (tf.nn.log_softmax(l_natural) - tf.nn.log_softmax(l_adv)))


def trades_loss(model, x, y, x_adv, beta):
    logits = model(x, train=True)
    loss_natural = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y, logits))
    loss_robust = criterion_kl(logits, model(x_adv, train=True)) / x.shape[0]
    train_loss = loss_natural + beta * loss_robust
    train_acc = tf.reduce_mean(tf.cast(tf.equal(y, tf.argmax(logits, axis=1)), dtype=tf.float32))

    return train_acc, train_loss


def trades_adv(model, x, eps, eps_iter, nb_iter, clip_range):
    logits_natural = model(x, train=False)
    x_adv = (x + 0.001 * tf.random.normal(x.shape, mean=0.0, stddev=1.0))
    for _ in range(nb_iter):
        with tf.GradientTape() as g:
            g.watch(x_adv)
            loss_kl = criterion_kl(logits_natural, model(x_adv, train=False))
        grad = g.gradient(loss_kl, x_adv)
        x_adv = x_adv + tf.multiply(eps_iter, tf.stop_gradient(tf.sign(grad)))
        x_adv = x + tf.clip_by_value(x_adv - x, -eps, eps)
        x_adv = tf.clip_by_value(x_adv, clip_range[0], clip_range[1])
    x_adv = tf.stop_gradient(tf.clip_by_value(x_adv, clip_range[0], clip_range[1]))

    return x_adv
