from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from ..model.lenet import LeNetConv, LeNetConvBayes
from ..model.attacks import FGSMAttack, PGDAttack
from ..model.vgg import VGG, VGGBayes
from ..model.misc import *
from ..utils.logger import Logger
from ..utils  import mnist, cifar10, cifar100
from ..utils.paths import RESULTS_PATH
import os
import re
import argparse
import random
from tqdm import tqdm

np.set_printoptions(precision=4, suppress=True)

# General flags
tf.flags.DEFINE_string("eval_mode", "train", "Which evaluation mode")
tf.flags.DEFINE_integer("gpuid", 0, "Which gpu id to use")

# Training flags
tf.flags.DEFINE_string("name", "test", "name of the model")
tf.flags.DEFINE_string("net", "lenet_fc", "Which net to use.")
tf.flags.DEFINE_string("mode", "base", "Which mode to use.")
tf.flags.DEFINE_string("data", "cifar10", "Which dataset to use.")
tf.flags.DEFINE_integer("batch_size", 100, "Batch size")
tf.flags.DEFINE_integer("save_freq", 20, "Saving frequency for model")
tf.flags.DEFINE_integer("n_epochs", 200, "No of epochs")
tf.flags.DEFINE_integer("directory", None, "Train directory")
tf.flags.DEFINE_integer("seed", None, "Random seed.")
tf.flags.DEFINE_float("kl_weight", 100.0, "Weight for kl term in bayesian network")
tf.flags.DEFINE_float("thres", 1e-3, "Threshold for dropout")
tf.flags.DEFINE_float("init_lr", 1e-2, "Learning rate")
tf.flags.DEFINE_float("gamma", None, "Gamma for vib cost")
tf.flags.DEFINE_boolean("adv_train", False, "Adversarial training or not")
tf.flags.DEFINE_boolean("train_source", False, "Train source model for black box")
tf.flags.DEFINE_boolean("sep_mask", False, "Seperate clean and adv mask")

# Attack flags
tf.flags.DEFINE_string("attack", "pgd", "Which attack to use.")
tf.flags.DEFINE_string("attack_source", "base", "Source model for attack.")
tf.flags.DEFINE_integer("pgd_steps", 40, "No of pgd steps")
tf.flags.DEFINE_float("adv_weight", 0.3, "Weight for adversarial cost")
tf.flags.DEFINE_float("eps", 0.03, "Epsilon for attack")
tf.flags.DEFINE_float("step_size", 0.007, "Step size for attack")
tf.flags.DEFINE_boolean("white_box", True, "White box/black box attack")

FLAGS = tf.app.flags.FLAGS

os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpuid)
os.environ['CUDA_CACHE_PATH'] = '/st2/divyam/tmp'

def init_random_seeds():
  tf.set_random_seed(FLAGS.seed)
  random.seed(FLAGS.seed)
  np.random.seed(FLAGS.seed)

if FLAGS.net.startswith('lenet_conv'):
    FLAGS.data = 'mnist'
    input_fn = mnist.input_fn
    NUM_TRAIN = mnist.NUM_TRAIN
    NUM_TEST = mnist.NUM_TEST
    n_classes = 10
    if FLAGS.train_source:
        net = LeNetConvBayes(name=FLAGS.net+'_source')  if FLAGS.net.endswith('bayes') \
            else LeNetConv(name=FLAGS.net+'_source')
    else:
        net = LeNetConvBayes() if FLAGS.net.endswith('bayes') \
            else LeNetConv()
    if (not FLAGS.white_box):
        source_net = LeNetConvBayes(name=FLAGS.net+'_source') if FLAGS.net.endswith('bayes') \
            else LeNetConv(name=FLAGS.net+'_source')

elif FLAGS.net.startswith('vgg'):
    vgg_mode = int(FLAGS.net.split('_')[0][-2:])
    if FLAGS.data == 'cifar10':
        input_fn = cifar10.input_fn
        NUM_TRAIN = cifar10.NUM_TRAIN
        NUM_TEST = cifar10.NUM_TEST
        n_classes = 10
    elif FLAGS.data == 'cifar100':
        input_fn = cifar100.input_fn
        NUM_TRAIN = cifar100.NUM_TRAIN
        NUM_TEST = cifar100.NUM_TEST
        n_classes = 100
    if FLAGS.train_source:
        net = VGGBayes(n_classes, mode=vgg_mode, name=FLAGS.net+'_source') if FLAGS.net.endswith('bayes') \
            else VGG(n_classes, mode=vgg_mode, name=FLAGS.net+'_source')
    else:
        net = VGGBayes(n_classes, mode=vgg_mode) if FLAGS.net.endswith('bayes') \
            else VGG(n_classes, mode=vgg_mode)
    if (not FLAGS.white_box):
        source_net = VGGBayes(n_classes, mode=vgg_mode, name=FLAGS.net+'_source') if FLAGS.net.endswith('bayes') \
            else VGG(n_classes, mode=vgg_mode, name=FLAGS.net+'_source')
else:
    raise ValueError('Invalid net {}'.format(FLAGS.net))


if FLAGS.directory is None:
    if FLAGS.adv_train:
        if FLAGS.train_source:
            directory = os.path.join(RESULTS_PATH, FLAGS.net, FLAGS.data, 'adv_train', 'attack_source', FLAGS.mode, FLAGS.name)
            base_directory = os.path.join(RESULTS_PATH, FLAGS.net, FLAGS.data, 'attack_source', 'base', FLAGS.name)
            bbd_directory = os.path.join(RESULTS_PATH, FLAGS.net, FLAGS.data, 'attack_source', 'bbd', FLAGS.name)
        else:
            directory = os.path.join(RESULTS_PATH, FLAGS.net, FLAGS.data, 'adv_train', 'default', FLAGS.mode, FLAGS.name)
            base_directory = os.path.join(RESULTS_PATH, FLAGS.net, FLAGS.data, 'default', 'base', FLAGS.name)
            bbd_directory = os.path.join(RESULTS_PATH, FLAGS.net, FLAGS.data, 'default', 'bbd', FLAGS.name)
    elif (not FLAGS.adv_train and FLAGS.train_source):
        directory = os.path.join(RESULTS_PATH, FLAGS.net, FLAGS.data, 'attack_source', FLAGS.mode, FLAGS.name)
        base_directory = os.path.join(RESULTS_PATH, FLAGS.net, FLAGS.data, 'attack_source', 'base', FLAGS.name)
        bbd_directory = os.path.join(RESULTS_PATH, FLAGS.net, FLAGS.data, 'attack_source', 'bbd', FLAGS.name)
    elif (not FLAGS.train_source):
        directory = os.path.join(RESULTS_PATH, FLAGS.net, FLAGS.data, 'default', FLAGS.mode, FLAGS.name)
        base_directory = os.path.join(RESULTS_PATH, FLAGS.net, FLAGS.data, 'default', 'base', FLAGS.name)
        bbd_directory = os.path.join(RESULTS_PATH, FLAGS.net, FLAGS.data, 'default', 'bbd', FLAGS.name)
else:
    directory = FLAGS.directory

if not FLAGS.white_box:
    # if FLAGS.adv_train:
    #     attack_source_directory = os.path.join(RESULTS_PATH, FLAGS.net, FLAGS.data, 'adv_train', 'attack_source', FLAGS.attack_source, FLAGS.name)
    # else:
    attack_source_directory = os.path.join(RESULTS_PATH, FLAGS.net, FLAGS.data, 'attack_source', FLAGS.attack_source, FLAGS.name)

x, y = input_fn(True, FLAGS.batch_size)
tx, ty = input_fn(False, FLAGS.batch_size)
n_train_batches = NUM_TRAIN // FLAGS.batch_size
n_test_batches = NUM_TEST // FLAGS.batch_size

if FLAGS.eval_mode == 'attack' or FLAGS.adv_train:
    x_clean = tf.placeholder(tf.float32, shape=x.shape)
    y_clean = tf.placeholder(tf.float32, shape=y.shape)
    x_adv = tf.placeholder(tf.float32, shape=x.shape)

    if FLAGS.attack == 'fgsm':
        if (not FLAGS.white_box):
            attack = FGSMAttack(x_clean, y_clean, source_net, FLAGS.attack_source, epsilon=FLAGS.eps)
        else:
            attack = FGSMAttack(x_clean, y_clean, net, FLAGS.mode, epsilon=FLAGS.eps)
    elif FLAGS.attack == 'pgd':
        if (not FLAGS.white_box):
            attack = PGDAttack(x_clean, y_clean, source_net, FLAGS.attack_source, epsilon=FLAGS.eps, num_steps=FLAGS.pgd_steps, step_size=FLAGS.step_size, random_start=True)
        else:
            attack = PGDAttack(x_clean, y_clean, net, FLAGS.mode, epsilon=FLAGS.eps, num_steps=FLAGS.pgd_steps, step_size=FLAGS.step_size, random_start=True)
    else:
        raise ValueError('Invalid attack {}'.format(FLAGS.attack))

def adv_train_model():
    print("Target mode", FLAGS.mode)
    print("Source mode", FLAGS.attack_source)

    if not os.path.isdir(directory):
        os.makedirs(directory)
    print ('results saved in {}'.format(directory))

    global_step = tf.train.get_or_create_global_step()
    sess = tf.Session()

    if FLAGS.sep_mask:
        # Train clean and adv accuracy using their own masks
        cent_clean, acc_clean = net.classify(x_clean, y_clean, mode=FLAGS.mode)
        cent_adv, acc_adv = net.classify(x_adv, y_clean, mode=FLAGS.mode+'_adv')

        # Train clean and adv accuracy using opposite masks
        cent_clean_adv, acc_clean_adv = net.classify(x_clean, y_clean, mode=FLAGS.mode+'_adv')
        cent_adv_clean, acc_adv_clean = net.classify(x_adv, y_clean, mode=FLAGS.mode)

        # Test clean and adv accuracy using their own masks
        tcent_clean, tacc_clean = net.classify(x_clean, y_clean, mode=FLAGS.mode, train=False)
        tcent_adv, tacc_adv = net.classify(x_adv, y_clean, mode=FLAGS.mode+'_adv', train=False)

        # Test clean and adv accuracy using their opposite masks
        tcent_clean_adv, tacc_clean_adv = net.classify(x_clean, y_clean, mode=FLAGS.mode+'_adv', train=False)
        tcent_adv_clean, tacc_adv_clean = net.classify(x_adv, y_clean, mode=FLAGS.mode, train=False)
    else:
        cent_clean, acc_clean = net.classify(x_clean, y_clean, mode=FLAGS.mode)
        cent_adv, acc_adv = net.classify(x_adv, y_clean, mode=FLAGS.mode)
        tcent_clean, tacc_clean = net.classify(x_clean, y_clean, mode=FLAGS.mode, train=False)
        tcent_adv, tacc_adv = net.classify(x_adv, y_clean, mode=FLAGS.mode, train=False)

    base_vars = net.params('base')
    base_trn_vars = net.params('base', trainable=True)

    if FLAGS.mode != 'base':
        mode_vars = net.params(FLAGS.mode, trainable=True)
        mode_adv_vars = net.params(FLAGS.mode+'_adv', trainable=True)
        kl = net.kl(mode=FLAGS.mode)
        if FLAGS.sep_mask:
            kl_adv = net.kl(mode=FLAGS.mode+'_adv')
        if FLAGS.mode == 'dbbd':
            n_active_x = net.n_active_x()
            bbd_vars = net.params('bbd')
            n_active = net.n_active(mode='bbd')
        else:
            n_active = net.n_active(mode=FLAGS.mode)
            if FLAGS.sep_mask:
                n_active_adv = net.n_active(mode=FLAGS.mode+'_adv')
                mask_diff = net.get_mask_diff(mode1=FLAGS.mode, mode2=FLAGS.mode+'_adv')

    if FLAGS.net.startswith('lenet_conv'):
        if FLAGS.mode == 'base':
            bdrs = [int(n_train_batches * FLAGS.n_epochs * r) for r in [.5, .7]]
            vals = [1e-3, 1e-4, 1e-5]
        else:
            bdrs = [int(n_train_batches * FLAGS.n_epochs * r) for r in [.5, .7]]
            vals1 = [FLAGS.init_lr * r for r in [1., 0.1, 0.01]]
            vals2 = [0.1 * v for v in vals1]
        gamma = 1e-5
    elif FLAGS.net.startswith('vgg'):
        if FLAGS.mode == 'base':
            bdrs = [int(n_train_batches * FLAGS.n_epochs * r) for r in [.5, .8]]
            vals = [1e-3, 1e-4, 1e-5]
        else:
            bdrs = [int(n_train_batches * FLAGS.n_epochs * r) for r in [.5, .8]]
            vals1 = [FLAGS.init_lr * r for r in [1., 0.1, 0.01]]
            vals2 = [0.1 * v for v in vals1]
        gamma = 1e-5

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    if FLAGS.mode == 'base':
        lr = get_staircase_lr(global_step, bdrs, vals)
        loss_mix = (cent_clean + FLAGS.adv_weight * cent_adv)/(1 + FLAGS.adv_weight) + 1e-4 * l2_loss(base_trn_vars)
        if FLAGS.net.endswith('bayes'):
            loss_mix = (cent_clean + FLAGS.adv_weight * cent_adv)/(1 + FLAGS.adv_weight)
            loss_mix += net.kl(mode=FLAGS.mode) / (NUM_TRAIN * FLAGS.kl_weight)
        with tf.control_dependencies(update_ops):
            train_op = tf.train.AdamOptimizer(lr).minimize(loss_mix, global_step=global_step)
        saver = tf.train.Saver(base_vars)
    else:
        lr1 = get_staircase_lr(global_step, bdrs, vals1)
        lr2 = get_staircase_lr(global_step, bdrs, vals2)
        if FLAGS.mode == 'vib':
            loss_mix = (FLAGS.adv_weight * cent_adv + cent_clean)/(1 + FLAGS.adv_weight) + kl * gamma + 1e-4 * l2_loss(base_trn_vars)
            clean_loss = cent_clean + kl*gamma + 1e-4 * l2_loss(base_trn_vars)
            adv_loss = cent_adv + kl_adv*gamma + 1e-4 * l2_loss(base_trn_vars)
            loss_mix = (FLAGS.adv_weight * cent_adv + cent_clean)/(1 + FLAGS.adv_weight)  + (kl*gamma + FLAGS.adv_weight * kl_adv*gamma)/(1 + FLAGS.adv_weight) + 1e-4 * l2_loss(base_trn_vars)
        else:
            #loss_mix = (FLAGS.adv_weight * cent_adv + cent_clean)/(1 + FLAGS.adv_weight)  + kl/NUM_TRAIN + 1e-4 * l2_loss(base_trn_vars)
            clean_loss = cent_clean + kl/NUM_TRAIN + 1e-4 * l2_loss(base_trn_vars)
            adv_loss = cent_adv + kl_adv/NUM_TRAIN + 1e-4 * l2_loss(base_trn_vars)
            loss_mix = (FLAGS.adv_weight * cent_adv + cent_clean)/(1 + FLAGS.adv_weight)  + (kl/NUM_TRAIN + FLAGS.adv_weight * kl_adv/NUM_TRAIN)/(1 + FLAGS.adv_weight) + 1e-4 * l2_loss(base_trn_vars)
        with tf.control_dependencies(update_ops):
            train_op1 = tf.train.AdamOptimizer(lr1).minimize(clean_loss,
                var_list=mode_vars, global_step=global_step)
            train_op1_adv = tf.train.AdamOptimizer(lr1).minimize(adv_loss,
                var_list=mode_adv_vars, global_step=global_step)
            train_op2 = tf.train.AdamOptimizer(lr2).minimize(loss_mix,
                var_list=base_trn_vars)
            train_op = tf.group(train_op1, train_op1_adv, train_op2)
        if FLAGS.mode == 'dbbd':
            all_vars = net.params()
            saver = tf.train.Saver(all_vars)
        else:
            saver = tf.train.Saver(base_vars + mode_vars + mode_adv_vars)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    if FLAGS.sep_mask:
        train_loss_summary = tf.summary.scalar("Clean train cent with clean mask", cent_clean)
        train_loss_adv_summary = tf.summary.scalar("Clean train cent with adv mask", cent_clean_adv)
        train_acc_summary = tf.summary.scalar("Clean train acc with clean mask", acc_clean)
        train_acc_adv_summary = tf.summary.scalar("Clean train acc with adv mask", acc_clean_adv)

        train_adv_loss_clean_summary = tf.summary.scalar("Adv train cent with clean mask", cent_adv_clean)
        train_adv_loss_summary = tf.summary.scalar("Adv train cent with adv mask", cent_adv)
        train_adv_acc_clean_summary = tf.summary.scalar("Adv train acc with clean mask", acc_adv_clean)
        train_adv_acc_summary = tf.summary.scalar("Adv train acc with adv mask", acc_adv)

        test_loss_summary = tf.summary.scalar("Clean test cent with clean mask", tcent_clean)
        test_loss_adv_summary = tf.summary.scalar("Clean test cent with adv mask", tcent_clean_adv)
        test_acc_summary = tf.summary.scalar("Clean test acc with clean mask", tacc_clean)
        test_acc_adv_summary = tf.summary.scalar("Clean test acc with adv mask", tacc_clean_adv)

        test_adv_loss_clean_summary = tf.summary.scalar("Adv test cent with clean mask", tcent_adv_clean)
        test_adv_loss_summary = tf.summary.scalar("Adv test cent with adv mask", tcent_adv)
        test_adv_acc_clean_summary = tf.summary.scalar("Adv test acc with clean mask", tacc_adv_clean)
        test_adv_acc_summary = tf.summary.scalar("Adv test acc with adv mask", tacc_adv)
    else:
        train_loss_summary = tf.summary.scalar("Clean train cent", cent_clean)
        train_acc_summary = tf.summary.scalar("Clean train acc", acc_clean)
        train_adv_loss_summary = tf.summary.scalar("Adv train cent", cent_adv)
        train_adv_acc_summary = tf.summary.scalar("Adv train acc", acc_adv)

        test_loss_summary = tf.summary.scalar("Clean test cent", tcent_clean)
        test_acc_summary = tf.summary.scalar("Clean test acc", tacc_clean)
        test_adv_loss_summary = tf.summary.scalar("Adv test cent", tcent_adv)
        test_adv_acc_summary = tf.summary.scalar("Adv test acc", tacc_adv)

    summary_op = tf.summary.merge_all()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(directory, sess.graph)
    ckpt = tf.train.get_checkpoint_state(directory)
    base_ckpt = tf.train.get_checkpoint_state(base_directory)
    start_epoch = 1
    if base_ckpt and base_ckpt.model_checkpoint_path and FLAGS.mode != 'base':
      print("Base model restored from: ", base_ckpt.model_checkpoint_path )
      tf.train.Saver(base_trn_vars).restore(sess, base_ckpt.model_checkpoint_path)
    if FLAGS.mode == 'dbbd':
        bbd_ckpt = tf.train.get_checkpoint_state(bbd_directory)
        print("BBD model restored from: ", bbd_ckpt.model_checkpoint_path)
        tf.train.Saver(base_trn_vars + bbd_vars).restore(sess, bbd_ckpt.model_checkpoint_path)
    if ckpt and ckpt.model_checkpoint_path:
        print("Model restored from: ", ckpt.model_checkpoint_path )
        tf.train.Saver(base_trn_vars).restore(sess, ckpt.model_checkpoint_path)
        global_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
        start_epoch = global_step//n_train_batches+1
        print(global_step, FLAGS.n_epochs)

    logfile = open(os.path.join(directory, 'train.log'), 'w')
    train_clean_logger = Logger('cent', 'acc')
    train_adv_logger = Logger('cent', 'acc')
    if FLAGS.sep_mask:
        train_clean_adv_logger = Logger('cent', 'acc')
        train_adv_clean_logger = Logger('cent', 'acc')

    if FLAGS.sep_mask:
        train_to_run = [train_op, cent_clean, acc_clean, cent_adv, acc_adv, cent_clean_adv, acc_clean_adv, cent_adv_clean, acc_adv_clean , summary_op]
    else:
        train_to_run = [train_op, cent_clean, acc_clean, cent_adv, acc_adv, summary_op]
    test_clean_logger = Logger('cent', 'acc')
    test_adv_logger = Logger('cent', 'acc')
    if FLAGS.sep_mask:
        test_clean_adv_logger = Logger('cent', 'acc')
        test_adv_clean_logger = Logger('cent', 'acc')

    if FLAGS.sep_mask:
        test_to_run = [tcent_clean, tacc_clean, tcent_adv, tacc_adv, tcent_clean_adv, tacc_clean_adv, tcent_adv_clean, tacc_adv_clean]
    else:
        test_to_run = [tcent_clean, tacc_clean, tcent_adv, tacc_adv]

    if FLAGS.mode == 'dbbd':
        test_to_run = [tcent_clean, tacc_clean, tcent_adv, tacc_adv, n_active_x]

    for epoch in range(start_epoch, (FLAGS.n_epochs+1)):
        if FLAGS.mode == 'base':
            line = 'Epoch {}, lr {:.3e}'.format(epoch, sess.run(lr))
            print(line)
            logfile.write(line+'\n')
        else:
            np_lr1, np_lr2 = sess.run([lr1, lr2])
            line = 'Epoch {}, {} lr {:.3e}, base lr {:.3e}'.format(epoch, FLAGS.mode, np_lr1, np_lr2)
            print(line)
            logfile.write(line+'\n')
        train_clean_logger.clear()
        train_adv_logger.clear()
        if FLAGS.sep_mask:
            train_clean_adv_logger.clear()
            train_adv_clean_logger.clear()

        for it in tqdm(range(1, n_train_batches+1)):
            x_np, y_np = sess.run([x, y])
            x_adv_np = attack.perturb(x_np, y_np, sess)
            x_adv_np = x_adv_np.astype(np.float32)
            if FLAGS.sep_mask:
                _, cent_clean, acc_clean, cent_adv, acc_adv, cent_clean_adv, acc_clean_adv, cent_adv_clean, acc_adv_clean, summary = sess.run(train_to_run, feed_dict={x_clean: x_np, x_adv: x_adv_np, y_clean: y_np})
            else:
                _, cent_clean, acc_clean, cent_adv, acc_adv, summary = sess.run(train_to_run, feed_dict={x_clean: x_np, x_adv: x_adv_np, y_clean: y_np})
            train_clean_logger.record([cent_clean, acc_clean])
            train_adv_logger.record([cent_adv, acc_adv])
            if FLAGS.sep_mask:
                train_clean_adv_logger.record([cent_clean_adv, acc_clean_adv])
                train_adv_clean_logger.record([cent_adv_clean, acc_adv_clean])

        writer.add_summary(summary, global_step=epoch)
        train_clean_logger.show(header='train_clean', epoch=epoch, logfile=logfile)
        train_adv_logger.show(header='train_adv', epoch=epoch, logfile=logfile)
        if FLAGS.sep_mask:
            train_clean_adv_logger.show(header='train_clean_adv', epoch=epoch, logfile=logfile)
            train_adv_clean_logger.show(header='train_adv_clean', epoch=epoch, logfile=logfile)

        test_clean_logger.clear()
        test_adv_logger.clear()
        if FLAGS.sep_mask:
            test_clean_adv_logger.clear()
            test_adv_clean_logger.clear()
        np_n_active_x = 0
        for it in range(1, n_test_batches+1):
            tx_clean_np, ty_clean_np = sess.run([tx, ty])
            tx_adv_np = attack.perturb(tx_clean_np, ty_clean_np, sess)
            tx_adv_np = tx_adv_np.astype(np.float32)

            res = sess.run(test_to_run, feed_dict={x_clean: tx_clean_np, y_clean: ty_clean_np, x_adv: tx_adv_np})

            if FLAGS.mode == 'dbbd':
                np_n_active_x += np.array(res[-1], dtype=float)
                test_clean_logger.record(res[:2])
                test_adv_logger.record(res[2:-1])
            else:
                test_clean_logger.record(res[:2])
                test_adv_logger.record(res[2:4])
                if FLAGS.sep_mask:
                    test_clean_adv_logger.record(res[4:6])
                    test_adv_clean_logger.record(res[6:])

        test_clean_logger.show(header='test_clean', epoch=epoch, logfile=logfile)
        test_adv_logger.show(header='test_adv', epoch=epoch, logfile=logfile)
        if FLAGS.sep_mask:
            test_clean_adv_logger.show(header='test_clean_adv', epoch=epoch, logfile=logfile)
            test_adv_clean_logger.show(header='test_adv_clean', epoch=epoch, logfile=logfile)

        if FLAGS.mode != 'base':
            if FLAGS.sep_mask:
                np_kl, np_n_active, np_adv_kl, np_n_active_adv, np_mask_diff = sess.run([kl, n_active, kl_adv, n_active_adv, mask_diff])
            np_kl, np_n_active = sess.run([kl, n_active])
            line = 'kl: ' + str(np_kl) + '\n'
            line += 'n_active: ' + str(np_n_active) + '\n'
            if FLAGS.sep_mask:
                line += 'kl_adv: ' + str(np_adv_kl) + '\n'
                line += 'n_adv_active: ' + str(np_n_active_adv)
                line += 'mask diff: ' + str(np_mask_diff)
            print()
            if FLAGS.mode == 'dbbd':
                np_n_active_x = (np_n_active_x/n_test_batches).astype(int)
                line += 'n_active_x: ' + str(np_n_active_x.tolist())
            print(line)
            logfile.write(line+'\n')
            print()
            logfile.write('\n')

        if epoch%FLAGS.save_freq == 0:
            saver.save(sess, os.path.join(directory, 'model'), global_step=global_step)
    logfile.close()
    saver.save(sess, os.path.join(directory, 'model'), global_step=global_step)


def train_model():
    if not os.path.isdir(directory):
        os.makedirs(directory)
    print ('results saved in {}'.format(directory))

    cent, acc = net.classify(x, y, mode=FLAGS.mode)
    tcent, tacc = net.classify(tx, ty, mode=FLAGS.mode, train=False)
    base_vars = net.params('base')
    base_trn_vars = net.params('base', trainable=True)

    if FLAGS.mode != 'base':
        mode_vars = net.params(FLAGS.mode, trainable=True)
        kl = net.kl(mode=FLAGS.mode)
        if FLAGS.mode == 'dbbd':
            n_active_x = net.n_active_x()
            bbd_vars = net.params('bbd', trainable=False)
            n_active = net.n_active(mode='bbd')
        else:
            n_active = net.n_active(mode=FLAGS.mode)

    global_step = tf.train.get_or_create_global_step()
    train_loss_summary = tf.summary.scalar("Training Cross entropy", cent)
    train_acc_summary = tf.summary.scalar("Training accuracy", acc)
    test_loss_summary = tf.summary.scalar("Test Cross entropy", tcent)
    test_acc_summary = tf.summary.scalar("Test accuracy", tacc)
    summary_op = tf.summary.merge_all()

    if FLAGS.net.startswith('lenet_conv'):
        if FLAGS.mode == 'base':
            bdrs = [int(n_train_batches * FLAGS.n_epochs * r) for r in [.5, .7]]
            vals = [1e-3, 1e-4, 1e-5]
        else:
            bdrs = [int(n_train_batches * FLAGS.n_epochs * r) for r in [.5, .7]]
            vals1 = [FLAGS.init_lr * r for r in [1., 0.1, 0.01]]
            vals2 = [0.1 * v for v in vals1]
        gamma = 1e-5
    elif FLAGS.net.startswith('vgg'):
        if FLAGS.mode == 'base':
            bdrs = [int(n_train_batches * FLAGS.n_epochs * r) for r in [.5, .8]]
            vals = [1e-3, 1e-4, 1e-5]
        else:
            bdrs = [int(n_train_batches * FLAGS.n_epochs * r) for r in [.5, .8]]
            vals1 = [FLAGS.init_lr * r for r in [1., 0.1, 0.01]]
            vals2 = [0.1 * v for v in vals1]
        gamma = 1e-5

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    if FLAGS.mode == 'base':
        lr = get_staircase_lr(global_step, bdrs, vals)
        l2 = l2_loss(base_trn_vars)
        loss = cent + 1e-4 * l2
        if FLAGS.net.endswith('bayes'):
            loss = cent
            kl = net.kl(FLAGS.mode) / (NUM_TRAIN * FLAGS.kl_weight)
            loss += kl
        with tf.control_dependencies(update_ops):
            train_op = tf.train.AdamOptimizer(lr).minimize(loss, global_step=global_step)
        saver = tf.train.Saver(base_vars)
    else:
        lr1 = get_staircase_lr(global_step, bdrs, vals1)
        lr2 = get_staircase_lr(global_step, bdrs, vals2)
        if FLAGS.mode == 'vib':
            loss = cent + kl * gamma + 1e-4 * l2_loss(base_trn_vars)
        else:
            loss = cent + kl/NUM_TRAIN + 1e-4 * l2_loss(base_trn_vars)
        with tf.control_dependencies(update_ops):
            train_op1 = tf.train.AdamOptimizer(lr1).minimize(loss,
                var_list=mode_vars, global_step=global_step)
            train_op2 = tf.train.AdamOptimizer(lr2).minimize(loss,
                var_list=base_trn_vars)
            train_op = tf.group(train_op1, train_op2)
        if FLAGS.mode == 'dbbd':
            all_vars = net.params()
            saver = tf.train.Saver(all_vars)
        else:
            saver = tf.train.Saver(base_vars + mode_vars)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(directory, sess.graph)
    ckpt = tf.train.get_checkpoint_state(directory)
    base_ckpt = tf.train.get_checkpoint_state(base_directory)
    start_epoch = 1
    if base_ckpt and base_ckpt.model_checkpoint_path:
      print("Base model restored from: ",base_ckpt.model_checkpoint_path )
      tf.train.Saver(base_trn_vars).restore(sess, base_ckpt.model_checkpoint_path)
    if FLAGS.mode == 'dbbd':
        bbd_ckpt = tf.train.get_checkpoint_state(bbd_directory)
        print("BBD model restored from: ", bbd_ckpt.model_checkpoint_path)
        tf.train.Saver(base_trn_vars + bbd_vars).restore(sess, bbd_ckpt.model_checkpoint_path)
    if ckpt and ckpt.model_checkpoint_path:
        print("Model restored from: ", ckpt.model_checkpoint_path )
        tf.train.Saver(base_trn_vars).restore(sess, ckpt.model_checkpoint_path)
        global_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
        start_epoch = global_step//n_train_batches+1
        print(global_step, FLAGS.n_epochs)

    logfile = open(os.path.join(directory, 'train.log'), 'w')
    train_logger = Logger('cent', 'acc')
    train_to_run = [train_op, cent, acc]
    test_logger = Logger('cent', 'acc')
    test_to_run = [tcent, tacc]
    if FLAGS.mode == 'dbbd':
        test_to_run = [tcent, tacc, n_active_x]

    for epoch in range(start_epoch, FLAGS.n_epochs+1):
        if FLAGS.mode == 'base':
            line = 'Epoch {}, lr {:.3e}'.format(epoch, sess.run(lr))
            print(line)
            logfile.write(line+'\n')
        else:
            np_lr1, np_lr2 = sess.run([lr1, lr2])
            line = 'Epoch {}, {} lr {:.3e}, base lr {:.3e}'.format(epoch, FLAGS.mode, np_lr1, np_lr2)
            print(line)
            logfile.write(line+'\n')
        train_logger.clear()
        for it in tqdm(range(1, n_train_batches + 1)):
            train_logger.record(sess.run(train_to_run))
            summary = sess.run(summary_op)

        writer.add_summary(summary, global_step=epoch)
        train_logger.show(header='train', epoch=epoch, logfile=logfile)

        test_logger.clear()
        np_n_active_x = 0
        for it in range(1, n_test_batches+1):
            res = sess.run(test_to_run)
            if FLAGS.mode == 'dbbd':
                np_n_active_x += np.array(res[-1], dtype=float)
                test_logger.record(res[:-1])
            else:
                test_logger.record(res)

        test_logger.show(header='test', epoch=epoch, logfile=logfile)
        if FLAGS.mode != 'base':
            np_kl, np_n_active = sess.run([kl, n_active])
            line = 'kl: ' + str(np_kl) + '\n'
            line += 'n_active: ' + str(np_n_active) + '\n'
            print()
            if FLAGS.mode == 'dbbd':
                np_n_active_x = (np_n_active_x/n_test_batches).astype(int)
                line += 'n_active_x: ' + str(np_n_active_x.tolist())
            print(line)
            logfile.write(line+'\n')
            print()
            logfile.write('\n')
        if epoch%FLAGS.save_freq == 0:
            saver.save(sess, os.path.join(directory, 'model'), global_step=global_step)
    logfile.close()
    saver.save(sess, os.path.join(directory, 'model'),global_step=global_step)

def test_model():
    print("Eval directory", directory)
    cent, acc = net.classify(tx, ty, mode=FLAGS.mode, train=False)

    sess = tf.Session()
    ckpt = tf.train.get_checkpoint_state(directory)
    if FLAGS.mode == 'base':
        if ckpt and ckpt.model_checkpoint_path:
            tf.train.Saver(net.params(FLAGS.mode)).restore(sess, ckpt.model_checkpoint_path)
    else:
        kl = net.kl(mode=FLAGS.mode)
        if ckpt and ckpt.model_checkpoint_path:
            if FLAGS.mode == 'dbbd':
                tf.train.Saver(net.params()).restore(
                    sess, ckpt.model_checkpoint_path)
            else:
                tf.train.Saver(net.params('base') + net.params(FLAGS.mode)).restore(
                    sess, ckpt.model_checkpoint_path)

        if FLAGS.mode == 'dbbd':
            n_active_x = net.n_active_x()
            n_active = net.n_active(mode='bbd')
        else:
            n_active = net.n_active(mode=FLAGS.mode)

    logger = Logger('cent', 'acc')
    np_n_active_x = 0
    for it in range(1, n_test_batches+1):
        logger.record(sess.run([cent, acc]))
        if FLAGS.mode == 'dbbd':
            np_n_active_x += np.array(sess.run(n_active_x), dtype=float)
    if FLAGS.mode != 'base':
        np_kl, np_n_active = sess.run([kl, n_active])
        print('kl: {:.4f}'.format(np_kl))
        print('n_active: ' + ' '.join(map(str, np_n_active)))
        if FLAGS.mode == 'dbbd':
            np_n_active_x = (np_n_active_x/n_test_batches).astype(int).tolist()
            print('n_active_x: ' + ' '.join(map(str, np_n_active_x)) + '\n')
    logger.show(header='test')


def attack_model():
    print("Target mode", FLAGS.mode)
    print("Source mode", FLAGS.attack_source)

    if FLAGS.sep_mask:
        # Test clean and adv accuracy using their own masks
        tcent_clean, tacc_clean = net.classify(x_clean, y_clean, mode=FLAGS.mode, train=False)
        tcent_adv, tacc_adv = net.classify(x_adv, y_clean, mode=FLAGS.mode+'_adv', train=False)

        # Test clean and adv accuracy using their opposite masks
        tcent_clean_adv, tacc_clean_adv = net.classify(x_clean, y_clean, mode=FLAGS.mode+'_adv', train=False)
        tcent_adv_clean, tacc_adv_clean = net.classify(x_adv, y_clean, mode=FLAGS.mode, train=False)
    else:
        tcent_clean, tacc_clean = net.classify(x_clean, y_clean, mode=FLAGS.mode, train=False)
        tcent_adv, tacc_adv = net.classify(x_adv, y_clean, mode=FLAGS.mode, train=False)

    sess = tf.Session()
    ckpt = tf.train.get_checkpoint_state(directory)
    print("Target for attack: ", directory)
    print("White box attack:", FLAGS.white_box)
    if FLAGS.mode == 'base':
        if ckpt and ckpt.model_checkpoint_path:
            tf.train.Saver(net.params(FLAGS.mode)).restore(sess, ckpt.model_checkpoint_path)
    else:
        kl = net.kl(mode=FLAGS.mode)
        if ckpt and ckpt.model_checkpoint_path:
            if FLAGS.mode == 'dbbd':
                tf.train.Saver(net.params()).restore(
                    sess, ckpt.model_checkpoint_path)
            else:
                tf.train.Saver(net.params('base') + net.params(FLAGS.mode)+net.params(FLAGS.mode+'_adv')).restore(
                    sess, ckpt.model_checkpoint_path)

        if FLAGS.mode == 'dbbd':
            n_active_x = net.n_active_x()
            n_active = net.n_active(mode='bbd')
        else:
            n_active = net.n_active(mode=FLAGS.mode)
            mask_diff = net.get_mask_diff(mode1=FLAGS.mode, mode2=FLAGS.mode+'_adv')

    if (not FLAGS.white_box):
        print("Source for attack: ", attack_source_directory)
        attack_source_ckpt = tf.train.get_checkpoint_state(attack_source_directory)
        if FLAGS.attack_source == "dbbd" or FLAGS.attack_source == "adv_dbbd":
            tf.train.Saver(source_net.params()).restore(
                    sess, attack_source_ckpt.model_checkpoint_path)
        elif FLAGS.attack_source != 'base':
            tf.train.Saver(source_net.params('base') + source_net.params(FLAGS.attack_source)).restore(
                    sess, attack_source_ckpt.model_checkpoint_path)
        else:
            tf.train.Saver(source_net.params('base')).restore(
                    sess, attack_source_ckpt.model_checkpoint_path)

    adv_logger = Logger('cent', 'acc')
    clean_logger = Logger('cent', 'acc')
    if FLAGS.sep_mask:
        clean_adv_logger = Logger('cent', 'acc')
        adv_clean_logger = Logger('cent', 'acc')
    for it in tqdm(range(1, n_test_batches + 1)):
        tx_clean_np, ty_clean_np = sess.run([tx, ty])
        tx_adv_np = attack.perturb(tx_clean_np, ty_clean_np, sess)
        tx_adv_np = tx_adv_np.astype(np.float32)

        #delta = tx_adv_np - tx_clean_np
        #linf_norm = np.amax(np.absolute(delta), axis=1)
        #min = np.amin(np.absolute(delta), axis=1)
        #percent_defended = np.sum((linf_norm < FLAGS.eps).astype(np.int32))/FLAGS.batch_size
        #print("Linf norm and Percent defended", linf_norm, percent_defended)
        if FLAGS.sep_mask:
            res = sess.run([tcent_clean, tacc_clean, tcent_adv, tacc_adv, tcent_clean_adv, tacc_clean_adv, tcent_adv_clean, tacc_adv_clean], feed_dict={x_clean: tx_clean_np, y_clean: ty_clean_np, x_adv: tx_adv_np})
        else:
            res = sess.run([tcent_clean, tacc_clean, tcent_adv, tacc_adv], feed_dict={x_clean: tx_clean_np, y_clean: ty_clean_np, x_adv: tx_adv_np})
        clean_logger.record(res[:2])
        adv_logger.record(res[2:4])
        if FLAGS.sep_mask:
            clean_adv_logger.record(res[4:6])
            adv_clean_logger.record(res[6:8])
    clean_logger.show(header='clean on clean mask')
    adv_logger.show(header='adv on adv mask')
    if FLAGS.sep_mask:
        clean_adv_logger.show(header='clean on adv mask')
        adv_clean_logger.show(header='adv on clean mask')
        print("Mask difference", sess.run(mask_diff))

def main(_):
    init_random_seeds()
    if (FLAGS.adv_train and FLAGS.eval_mode == 'train'):
        adv_train_model()
    elif (not FLAGS.adv_train and FLAGS.eval_mode == 'train'):
        train_model()
    elif FLAGS.eval_mode == 'test':
        test_model()
    elif FLAGS.eval_mode == 'attack':
        attack_model()
    else:
        raise ValueError('Invalid mode {}'.format(FLAGS.eval_mode))

if __name__=='__main__':
    tf.app.run()
