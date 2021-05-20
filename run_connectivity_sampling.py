import numpy as np
import time
import os
import pickle
import argparse
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from utils import *


def main():
    parser = argparse.ArgumentParser(description='End-to-end robust adversarial training of sparse neural networks')
    parser.add_argument('--data', type=str, default="cifar10", help='dataset for classification')
    parser.add_argument('--n_classes', type=int, default=10, help='number of classes in dataset')
    parser.add_argument('--model', type=str, default="vgg16", help='network architecture')
    parser.add_argument('--objective', type=str, default="at", help='options: natural, at, mat, trades, mart, rst')
    parser.add_argument('--epsilon', type=float, default=8/255, help='total perturbation epsilon during training')
    parser.add_argument('--eps_iter', type=float, default=2/255, help='per-iteration epsilon during adv. training')
    parser.add_argument('--num_iter', type=int, default=10, help='number of iterations for adv. training')
    parser.add_argument('--rst_frac', type=float, default=0.5, help='fraction of pseudo-labeled train samples for RST')
    parser.add_argument('--beta', type=float, default=6.0, help='robust loss regularizer weight (for TRADES or MART)')
    parser.add_argument('--n_epochs', '-e', type=int, default=200, help='number of training epochs')
    parser.add_argument('--batch', '-b', type=int, default=128, help='training mini-batch size')
    parser.add_argument('--l_rate', '-lr', type=float, default=1e-1, help='initial learning rate')
    parser.add_argument('--w_decay', '-wd', type=float, default=5e-4, help='initial weight decay factor')
    parser.add_argument('--noise_scaling', type=float, default=1e-6, help='noise scaling factor parameter')
    parser.add_argument('--sparse_train', '-s', action='store_true', help='perform robust end-to-end sparse training')
    parser.add_argument('--connectivity', '-pc', type=float, default=0.01, help='sparse connectivity (default: 1%)')
    parser.add_argument('--load_model', action='store_true', help='load saved weights for evaluation')
    args = parser.parse_args()

    # Load model and data components, retrieve/create useful variables
    model = model_loader(args)
    (x_train, y_train), (x_test, y_test), pix_range = data_loader(args)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(args.batch)
    aux_batch_size = int(args.batch * args.rst_frac) if args.objective == "rst" else 0
    train_batch_size = int(args.batch - aux_batch_size)
    iter_per_epoch = int(x_train.shape[0] / train_batch_size)
    lr_fn, wd_fn, optimizer = scheduler_loader(args, iter_per_epoch)
    filename = get_filename(args)
    train_results = {'epoch_l': [], 'iter_l': [], 'loss_l': [], 'acc_l': [], 'train_time_l': []}
    test_results = {'epoch_l': [], 'iter_l': [], 'loss_l': [], 'acc_l': [], 'test_time_l': []}

    def loss_fn(logits_predict, y_true):
        x_ent = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=logits_predict))
        is_correct = tf.equal(y_true, tf.argmax(logits_predict, axis=1))
        accuracy = tf.reduce_mean(tf.cast(is_correct, dtype=tf.float32))
        return accuracy, x_ent, is_correct

    def evaluate(ds):
        test_loss, test_correct, test_len = [], [], []
        for (test_batch_x, test_batch_y) in ds:
            _, loss, corr = loss_fn(model(test_batch_x, train=False), test_batch_y)
            test_loss.append(loss)
            test_correct.append(tf.reduce_sum(tf.cast(corr, tf.int32)))
            test_len.append(len(corr))
        test_acc = sum(test_correct) / sum(test_len)
        test_loss = sum(test_loss) / len(test_loss)
        return test_acc, test_loss

    def test_step():
        t0 = time.time()
        test_acc, test_loss = evaluate(test_ds)
        test_time = time.time() - t0

        if args.sparse_train:
            global_conn, layer_conn, num_conn = model.connectivity_stats()
            print('Epoch/iter: {}/{}\t time: [train: {:.3g}s / test: {:.3g}s]\t test acc: {:.3g}\t test loss: {:.3g}'
                  '\t connectivity: {:.3g}%\t #connections: '.
                  format(e + 1, k, train_results['train_time_l'][-1], test_time, test_acc, test_loss, global_conn) +
                  np.array_str(num_conn) + '\t layer wise: ' + np.array_str(np.array(layer_conn), precision=3))
        else:
            print('Epoch/iter: {}/{}\t time: [train: {:.3g}s / test: {:.3g}s]\t test acc: {:.3g}\t test loss: {:.3g}'.
                  format(e + 1, k, train_results['train_time_l'][-1], test_time, test_acc, test_loss))

        for key, variable in zip(['epoch', 'iter', 'loss', 'acc', 'test_time'], [e + 1, k, test_loss, test_acc, test_time]):
            test_results[key + '_l'].append(variable)

        return test_acc

    def train_step(x, y, x_adv=None):
        l_rate = lr_fn(optimizer.iterations.numpy())
        t0 = time.time()
        with tf.GradientTape() as loss_tape:
            if args.objective == "trades" or args.objective == "rst":
                train_acc, train_loss = trades_loss(model, x, y, x_adv, beta=args.beta)
            elif args.objective == "mart":
                train_acc, train_loss = mart_loss(model, x, y, x_adv, beta=args.beta)
            else:
                logits = model(x, train=True)
                train_acc, train_loss, _ = loss_fn(logits, y)
        grads = loss_tape.gradient(train_loss, model.param_list(trainable=True))

        if args.sparse_train:
            connectivity_matrix = [tf.cast(tf.greater(th, 0), tf.float32) for th in model.rewiring_param_list()]
            optimizer.apply_gradients(zip(grads, model.param_list(trainable=True)))
            [th.assign_add(l_rate * tf.random.normal(stddev=args.noise_scaling, shape=tf.shape(th)) *
                           connectivity_matrix[i]) for i, th in enumerate(model.rewiring_param_list())]
            model.sample_connectivity()
        else:
            optimizer.apply_gradients(zip(grads, model.param_list(trainable=True)))
        train_time = time.time() - t0

        for key, variable in zip(['epoch', 'iter', 'loss', 'acc', 'train_time'], [e + 1, k, train_loss, train_acc, train_time]):
            train_results[key + '_l'].append(variable)

    # Load saved model weights or evaluate clean test accuracy
    if args.load_model:
        w_dict = pickle.load(open(os.path.join('results', filename + '_best_weights.pickle'), 'rb'))
        [th.assign(w_dict['param_list'][i]) for i, th in enumerate(model.param_list(trainable=False))]
    else:
        # Create data augmentation generator/iterators
        train_gen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
        aux_iterator = data_aux_loader(aux_batch_size) if args.objective == "rst" else None

        # Main training loop
        best_te_acc = 0
        for e in range(args.n_epochs):
            for k, (batch_xs, batch_ys) in enumerate(train_gen.flow(x_train, y_train, shuffle=True, batch_size=train_batch_size)):
                if args.objective == "natural":
                    train_step(batch_xs, batch_ys)
                elif args.objective == "at":
                    batch_xs_adv = projected_gradient_descent(model, batch_xs, args.epsilon, pix_range, batch_ys,
                                                              args.num_iter, args.eps_iter)
                    train_step(batch_xs_adv, batch_ys)
                elif args.objective == "mat":
                    batch_xs_half = projected_gradient_descent(model, batch_xs[:(train_batch_size // 2)], args.epsilon,
                                                               pix_range, batch_ys[:(train_batch_size // 2)],
                                                               args.num_iter, args.eps_iter)
                    batch_xs_adv = tf.concat([batch_xs_half, batch_xs[(train_batch_size // 2):]], axis=0, type=tf.float32)
                    train_step(batch_xs_adv, batch_ys)
                elif args.objective == "trades":
                    batch_xs_adv = trades_adv(model, batch_xs, args.epsilon, args.eps_iter, args.num_iter, pix_range)
                    train_step(batch_xs, batch_ys, batch_xs_adv)
                elif args.objective == "mart":
                    batch_xs_adv = mart_adv(model, batch_xs, batch_ys, args.epsilon, args.eps_iter, args.num_iter, pix_range)
                    train_step(batch_xs, batch_ys, batch_xs_adv)
                elif args.objective == "rst":
                    batch_xs_aux, batch_ys_aux = aux_iterator.next()
                    batch_xs, batch_ys = tf.concat([batch_xs, batch_xs_aux], axis=0), tf.concat([batch_ys, batch_ys_aux], axis=0)
                    batch_xs_adv = trades_adv(model, batch_xs, args.epsilon, args.eps_iter, args.num_iter, pix_range)
                    train_step(batch_xs, batch_ys, batch_xs_adv)
                else:
                    raise NotImplementedError

                if k == 0:
                    te_acc = test_step()
                    if te_acc > best_te_acc:
                        best_te_acc = te_acc
                        best = {'param_list': model.param_list(trainable=False)}
                        pickle.dump(best, open(os.path.join('results', filename + '_best_weights.pickle'), 'wb'),
                                    protocol=pickle.HIGHEST_PROTOCOL)
                    else:
                        pass

                if k + 1 >= iter_per_epoch:
                    break
        _ = test_step()

        # Save results
        w_dict = {'param_list': model.param_list(trainable=False)}
        pickle.dump(w_dict, open(os.path.join('results', filename + '_weights.pickle'), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(train_results, open(os.path.join('results', filename + '_train_results.pickle'), 'wb'))
        pickle.dump(test_results, open(os.path.join('results', filename + '_test_results.pickle'), 'wb'))

    # Final test set evaluations
    intact_acc, _ = evaluate(test_ds)
    print('accuracy on clean examples (%): {:.3f}'.format(intact_acc.numpy() * 100))


if __name__ == '__main__':
    main()
