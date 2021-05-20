import tensorflow as tf


def keras_model_loader(args, p_list):
    keras_model = net(arch=args.model, n_classes=args.n_classes)
    if not args.sparse_train:
        p_list = list(map(lambda x: x.numpy(), p_list))
        keras_model.set_weights(p_list)
    else:
        def get_weight(theta, sign):
            return tf.where(condition=tf.greater(theta, 0),
                            x=tf.multiply(sign, theta),
                            y=tf.zeros(sign.shape, dtype=tf.float32))
        list_counter = 0
        weight_list = []
        while list_counter < len(p_list):
            layer = p_list[list_counter]
            if layer.name.find("/T:") == -1:
                weight_list.append(layer.numpy())
                list_counter += 1
            else:  # for layers with "theta", there will be a "sign" as well. combine these to get the exact weights.
                next_layer = p_list[list_counter + 1]
                weight_list.append(get_weight(layer, next_layer).numpy())
                list_counter += 2
        keras_model.set_weights(weight_list)
    return keras_model


def net(arch='vgg16', n_classes=10):
    if arch == 'vgg16':
        model = vgg16(n_classes)
    elif arch == 'resnet18':
        model = resnet18(n_classes)
    elif arch == 'resnet34':
        model = resnet34(n_classes)
    elif arch == 'resnet50':
        model = resnet50(n_classes)
    elif arch == 'wrn28_2':
        model = wideresnet28_2(n_classes)
    elif arch == 'wrn28_4':
        model = wideresnet28_4(n_classes)
    elif arch == 'wrn28_10':
        model = wideresnet28_10(n_classes)
    elif arch == 'wrn34_10':
        model = wideresnet34_10(n_classes)
    else:
        raise NotImplementedError
    return model


def wideresnet28_2(num_classes):
    model = wideresnet(num_classes, depth=28, widen_factor=2)
    return model


def wideresnet28_4(num_classes):
    model = wideresnet(num_classes, depth=28, widen_factor=4)
    return model


def wideresnet28_10(num_classes):
    model = wideresnet(num_classes, depth=28, widen_factor=10)
    return model


def wideresnet34_10(num_classes):
    model = wideresnet(num_classes, depth=34, widen_factor=10)
    return model


def resnet18(num_classes):
    model = resnet(num_classes, 'residual', [2, 2, 2, 2])
    return model


def resnet34(num_classes):
    model = resnet(num_classes, 'residual', [3, 4, 6, 3])
    return model


def resnet50(num_classes):
    model = resnet(num_classes, 'bottleneck', [3, 4, 6, 3])
    return model


def basic_residual(input, in_units, units, stride):
    expansion = 1

    h = tf.keras.layers.Conv2D(units, kernel_size=3, strides=stride, padding='SAME', use_bias=False)(input)
    h = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(h)
    h = tf.keras.layers.Activation(tf.keras.activations.relu)(h)
    h = tf.keras.layers.Conv2D(units, kernel_size=3, strides=1, padding='SAME', use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(h)
    h = tf.keras.layers.Activation(tf.keras.activations.linear)(h)  # dummy pass through to fix weight ordering of Keras

    if stride != 1 or in_units != expansion * units:
        input = tf.keras.layers.Conv2D(expansion * units, kernel_size=1, strides=stride, padding='VALID',
                                       use_bias=False)(input)
        input = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(input)
    h = tf.keras.layers.Add()([h, input])
    h = tf.keras.layers.Activation(tf.keras.activations.relu)(h)

    return h


def bottleneck(input, in_units, units, stride):
    expansion = 4

    h = tf.keras.layers.Conv2D(units, kernel_size=1, strides=1, padding='VALID', use_bias=False)(input)
    h = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(h)
    h = tf.keras.layers.Activation(tf.keras.activations.relu)(h)
    h = tf.keras.layers.Conv2D(units, kernel_size=3, strides=stride, padding='SAME', use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(h)
    h = tf.keras.layers.Activation(tf.keras.activations.relu)(h)
    h = tf.keras.layers.Conv2D(expansion * units, kernel_size=1, strides=1, padding='VALID', use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(h)
    h = tf.keras.layers.Activation(tf.keras.activations.linear)(h)  # dummy pass through to fix weight ordering of Keras

    if stride != 1 or in_units != expansion * units:
        input = tf.keras.layers.Conv2D(expansion * units, kernel_size=1, strides=stride, padding='VALID',
                                       use_bias=False)(input)
        input = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(input)
    h = tf.keras.layers.Add()([h, input])
    h = tf.keras.layers.Activation(tf.keras.activations.relu)(h)

    return h


def basic_block(input, in_units, units, stride):
    equalInOut = (in_units == units)

    if not equalInOut:
        input = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(input)
        input = tf.keras.layers.Activation(tf.keras.activations.relu)(input)
        z = tf.keras.layers.Conv2D(units, kernel_size=3, strides=stride, padding='SAME', use_bias=False)(input)
    else:
        z = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(input)
        z = tf.keras.layers.Activation(tf.keras.activations.relu)(z)
        z = tf.keras.layers.Conv2D(units, kernel_size=3, strides=stride, padding='SAME', use_bias=False)(z)

    z = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(z)
    z = tf.keras.layers.Activation(tf.keras.activations.relu)(z)
    z = tf.keras.layers.Conv2D(units, kernel_size=3, strides=1, padding='SAME', use_bias=False)(z)

    if equalInOut:
        z = tf.keras.layers.Add()([z, input])
    else:
        input = tf.keras.layers.Conv2D(units, kernel_size=1, strides=stride, padding='VALID', use_bias=False)(input)
        z = tf.keras.layers.Add()([z, input])

    return z


def network_block(input, nb_layers, in_units, units, stride):
    z = basic_block(input, in_units, units, stride)
    for i in range(1, int(nb_layers)):
        z = basic_block(z, units, units, 1)
    return z


def wideresnet(num_classes, depth, widen_factor):
    units = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
    n = (depth - 4) / 6

    x = tf.keras.Input(shape=(32, 32, 3))
    z = tf.keras.layers.Conv2D(units[0], kernel_size=3, strides=1, padding='SAME', use_bias=False)(x)
    z = network_block(z, n, units[0], units[1], stride=1)
    z = network_block(z, n, units[1], units[2], stride=2)
    z = network_block(z, n, units[2], units[3], stride=2)
    z = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(z)
    z = tf.keras.layers.Activation(tf.keras.activations.relu)(z)
    z = tf.keras.layers.AveragePooling2D(pool_size=8, strides=8)(z)
    z = tf.keras.layers.Flatten()(z)
    output = tf.keras.layers.Dense(num_classes, use_bias=True)(z)
    model = tf.keras.Model(inputs=x, outputs=output)

    return model


def resnet(num_classes, block, num_blocks):
    in_units = 64

    x = tf.keras.Input(shape=(32, 32, 3))
    z = tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, padding='SAME', use_bias=False)(x)
    z = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(z)
    z = tf.keras.layers.Activation(tf.keras.activations.relu)(z)
    for i, tup in enumerate([(64, 1), (128, 2), (256, 2), (512, 2)]):
        strides = [tup[1]] + [1] * (num_blocks[i] - 1)
        for st in strides:
            if block == 'residual':
                expansion = 1
                z = basic_residual(z, in_units=in_units, units=tup[0], stride=st)
                in_units = tup[0] * expansion
            elif block == 'bottleneck':
                expansion = 4
                z = bottleneck(z, in_units=in_units, units=tup[0], stride=st)
                in_units = tup[0] * expansion
            else:
                raise NotImplementedError
    z = tf.keras.layers.AveragePooling2D(pool_size=4, strides=4)(z)
    z = tf.keras.layers.Flatten()(z)
    output = tf.keras.layers.Dense(num_classes, use_bias=True)(z)
    model = tf.keras.Model(inputs=x, outputs=output)

    return model


def vgg16(num_classes):
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(32, 32, 3)))

    model.add(tf.keras.layers.Conv2D(64, kernel_size=3, padding='SAME', use_bias=False, name='vgg16/conv1'))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='vgg16/bn1'))
    model.add(tf.keras.layers.Activation(tf.keras.activations.relu))
    model.add(tf.keras.layers.Conv2D(64, kernel_size=3, padding='SAME', use_bias=False, name='vgg16/conv2'))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='vgg16/bn2'))
    model.add(tf.keras.layers.Activation(tf.keras.activations.relu))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides=2))

    model.add(tf.keras.layers.Conv2D(128, kernel_size=3, padding='SAME', use_bias=False, name='vgg16/conv3'))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='vgg16/bn3'))
    model.add(tf.keras.layers.Activation(tf.keras.activations.relu))
    model.add(tf.keras.layers.Conv2D(128, kernel_size=3, padding='SAME', use_bias=False, name='vgg16/conv4'))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='vgg16/bn4'))
    model.add(tf.keras.layers.Activation(tf.keras.activations.relu))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides=2))

    model.add(tf.keras.layers.Conv2D(256, kernel_size=3, padding='SAME', use_bias=False, name='vgg16/conv5'))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='vgg16/bn5'))
    model.add(tf.keras.layers.Activation(tf.keras.activations.relu))
    model.add(tf.keras.layers.Conv2D(256, kernel_size=3, padding='SAME', use_bias=False, name='vgg16/conv6'))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='vgg16/bn6'))
    model.add(tf.keras.layers.Activation(tf.keras.activations.relu))
    model.add(tf.keras.layers.Conv2D(256, kernel_size=3, padding='SAME', use_bias=False, name='vgg16/conv7'))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='vgg16/bn7'))
    model.add(tf.keras.layers.Activation(tf.keras.activations.relu))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides=2))

    model.add(tf.keras.layers.Conv2D(512, kernel_size=3, padding='SAME', use_bias=False, name='vgg16/conv8'))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='vgg16/bn8'))
    model.add(tf.keras.layers.Activation(tf.keras.activations.relu))
    model.add(tf.keras.layers.Conv2D(512, kernel_size=3, padding='SAME', use_bias=False, name='vgg16/conv9'))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='vgg16/bn9'))
    model.add(tf.keras.layers.Activation(tf.keras.activations.relu))
    model.add(tf.keras.layers.Conv2D(512, kernel_size=3, padding='SAME', use_bias=False, name='vgg16/conv10'))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='vgg16/bn10'))
    model.add(tf.keras.layers.Activation(tf.keras.activations.relu))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides=2))

    model.add(tf.keras.layers.Conv2D(512, kernel_size=3, padding='SAME', use_bias=False, name='vgg16/conv11'))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='vgg16/bn11'))
    model.add(tf.keras.layers.Activation(tf.keras.activations.relu))
    model.add(tf.keras.layers.Conv2D(512, kernel_size=3, padding='SAME', use_bias=False, name='vgg16/conv12'))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='vgg16/bn12'))
    model.add(tf.keras.layers.Activation(tf.keras.activations.relu))
    model.add(tf.keras.layers.Conv2D(512, kernel_size=3, padding='SAME', use_bias=False, name='vgg16/conv13'))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='vgg16/bn13'))
    model.add(tf.keras.layers.Activation(tf.keras.activations.relu))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, use_bias=True, name='vgg16/dense14'))
    model.add(tf.keras.layers.Activation(tf.keras.activations.relu))
    model.add(tf.keras.layers.Dense(256, use_bias=True, name='vgg16/dense15'))
    model.add(tf.keras.layers.Activation(tf.keras.activations.relu))
    model.add(tf.keras.layers.Dense(num_classes, use_bias=True, name='vgg16/dense16'))

    return model
