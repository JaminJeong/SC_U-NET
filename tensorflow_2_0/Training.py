#!/usr/bin/env python
# coding: utf-8

# ##### Copyright 2019 The TensorFlow Authors.
# 
# Licensed under the Apache License, Version 2.0 (the "License");

#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import absolute_import, division, print_function, unicode_literals

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="1"

try:
  # %tensorflow_version only exists in Colab.
  get_ipython().run_line_magic('tensorflow_version', '2.x')
except Exception:
  pass
import tensorflow as tf

import os
import time
from config import *
from util import *

from U_net import Generator

generator = Generator()

def generator_loss(gen_output, target):
  # mean absolute error
  l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

  return l1_loss


loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)


checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 generator=generator)

import datetime
log_dir="logs/"

summary_writer = tf.summary.create_file_writer(
  log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))


@tf.function
def train_step(input_image, target, epoch):
  with tf.GradientTape() as gen_tape:
    gen_output = generator(input_image, training=True)

    gen_l1_loss = generator_loss(gen_output, target)

  generator_gradients = gen_tape.gradient(gen_l1_loss,
                                          generator.trainable_variables)
  generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
  with summary_writer.as_default():
    tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)


# The actual training loop:
# 
# * Iterates over the number of epochs.
# * On each epoch it clears the display, and runs `generate_images` to show it's progress.
# * On each epoch it iterates over the training dataset, printing a '.' for each example.
# * It saves a checkpoint every 20 epochs.

def fit(train_ds, epochs):
  for epoch in range(epochs):
    start = time.time()
    print("Epoch: ", epoch)

    # Train
    for n, (input_image, target) in enumerate(train_ds):
      print('.', end='')
      if (n+1) % 100 == 0:
        print()
      train_step(input_image, target, epoch)
    print()

    # saving (checkpoint) the model every 20 epochs
    if (epoch + 1) % 20 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                        time.time()-start))
  checkpoint.save(file_prefix = checkpoint_prefix)

from tfrecords import load_tfrecords

dataset = load_tfrecords('tensorflow_2_0/sound_seperator.tfrecords')

# X_list, Y_list = LoadSpectrogram(".")  # Mix spectrogram
# X_mag, X_phase = Magnitude_phase_x(X_list)
# Y_mag, _ = Magnitude_phase_y(Y_list)
# X, Y = sampling(X_mag, Y_mag)
dataset_X = tf.data.Dataset.from_tensor_slices(X)
dataset_Y = tf.data.Dataset.from_tensor_slices(Y)
train_dataset = tf.data.Dataset.zip(zip(dataset_X, dataset_Y))
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH)

# Now run the training loop:
fit(training_set, EPOCHS)