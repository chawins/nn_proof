from __future__ import absolute_import

from datetime import datetime
import json
import os
import shutil
from timeit import default_timer as timer

import tensorflow as tf
import numpy as np

from lib.utils import load_gtsrb
from stn.conv_model import conv_model_no_color_adjust, build_cnn_no_stn
from pgd_attack import LinfPGDAttack

with open('config.json') as config_file:
    config = json.load(config_file)

# Setting up training parameters
tf.set_random_seed(config['random_seed'])

max_num_training_steps = config['max_num_training_steps']
num_output_steps = config['num_output_steps']
num_summary_steps = config['num_summary_steps']
num_checkpoint_steps = config['num_checkpoint_steps']

batch_size = config['training_batch_size']

# Load data
X_train, y_train, X_val, y_val, X_test, y_test = load_gtsrb()

# Set u
global_step = tf.contrib.framework.get_or_create_global_step()
model = build_cnn_no_stn()

# Setting up the optimizer
self.x_input = tf.placeholder(tf.float32, shape=[None, 784])
self.y_input = tf.placeholder(tf.int64, shape=[None])

self.x_image = tf.reshape(self.x_input, [-1, 28, 28, 1])

train_step = tf.train.AdamOptimizer(1e-4).minimize(loss,
                                                   global_step=global_step)

# Set up adversary
attack = LinfPGDAttack(model,
                       config['epsilon'],
                       config['k'],
                       config['a'],
                       config['random_start'],
                       config['loss_func'])

# Setting up the Tensorboard and checkpoint outputs
model_dir = config['model_dir']
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# We add accuracy and xent twice so we can easily make three types of
# comparisons in Tensorboard:
# - train vs eval (for a single run)
# - train of different runs
# - eval of different runs

saver = tf.train.Saver(max_to_keep=3)
tf.summary.scalar('accuracy adv train', model.accuracy)
tf.summary.scalar('accuracy adv', model.accuracy)
tf.summary.scalar('xent adv train', model.xent / batch_size)
tf.summary.scalar('xent adv', model.xent / batch_size)
tf.summary.image('images adv train', model.x_image)
merged_summaries = tf.summary.merge_all()

shutil.copy('config.json', model_dir)

with K.get_session() as sess:
    # Initialize the summary writer, global variables, and our time counter.
    summary_writer = tf.summary.FileWriter(model_dir, sess.graph)
    sess.run(tf.global_variables_initializer())
    training_time = 0.0

    # Main training loop
    for ii in range(max_num_training_steps):
        x_batch, y_batch = mnist.train.next_batch(batch_size)

        # Compute Adversarial Perturbations
        start = timer()
        x_batch_adv = attack.perturb(x_batch, y_batch, sess)
        end = timer()
        training_time += end - start

        nat_dict = {model.x_input: x_batch,
                    model.y_input: y_batch}

        adv_dict = {model.x_input: x_batch_adv,
                    model.y_input: y_batch}

        # Output to stdout
        if ii % num_output_steps == 0:
            nat_acc = sess.run(model.accuracy, feed_dict=nat_dict)
            adv_acc = sess.run(model.accuracy, feed_dict=adv_dict)
            print('Step {}:    ({})'.format(ii, datetime.now()))
            print('    training nat accuracy {:.4}%'.format(nat_acc * 100))
            print('    training adv accuracy {:.4}%'.format(adv_acc * 100))
            if ii != 0:
                print('    {} examples per second'.format(
                    num_output_steps * batch_size / training_time))
                training_time = 0.0
        # Tensorboard summaries
        if ii % num_summary_steps == 0:
            summary = sess.run(merged_summaries, feed_dict=adv_dict)
            summary_writer.add_summary(summary, global_step.eval(sess))

        # Write a checkpoint
        if ii % num_checkpoint_steps == 0:
            saver.save(sess,
                       os.path.join(model_dir, 'checkpoint'),
                       global_step=global_step)

        # Actual training step
        start = timer()
        sess.run(train_step, feed_dict=adv_dict)
        end = timer()
        training_time += end - start