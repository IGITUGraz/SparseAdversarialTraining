import tensorflow as tf


# Only considers l_infinity norm FGSM attack
def fast_gradient_sign_method(net, x, eps, clip_range, y, targeted=False):
    loss_fn = tf.nn.sparse_softmax_cross_entropy_with_logits
    with tf.GradientTape() as g:
        g.watch(x)
        loss = loss_fn(labels=y, logits=net(x, train=False))
        if targeted:
            loss = -loss
    grad = g.gradient(loss, x)

    optimal_perturbation = tf.multiply(eps, tf.stop_gradient(tf.sign(grad)))
    adv_x = x + optimal_perturbation
    adv_x = tf.clip_by_value(adv_x, clip_range[0], clip_range[1])
    return tf.cast(adv_x, tf.float32)


# Only considers l_infinity norm PGD attack
def projected_gradient_descent(net, x, eps, clip_range, y, nb_iter, eps_iter, rand_step=True, targeted=False):
    eta = tf.random.uniform(x.shape, -eps, eps) if rand_step else tf.zeros_like(x)
    adv_x = x + eta
    adv_x = tf.clip_by_value(adv_x, clip_range[0], clip_range[1])

    for _ in range(nb_iter):
        adv_x = fast_gradient_sign_method(net, adv_x, eps_iter, clip_range, y=y, targeted=targeted)
        adv_x = x + clip_perturbation_to_norm_ball(adv_x - x, eps)
        adv_x = tf.clip_by_value(adv_x, clip_range[0], clip_range[1])
    return tf.cast(adv_x, tf.float32)


# Only considers l_infinity norm ball
def clip_perturbation_to_norm_ball(perturbation, eps):
    return tf.clip_by_value(perturbation, -eps, eps)
