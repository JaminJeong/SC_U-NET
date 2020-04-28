import tensorflow as tf
from config import *

def conv2d(filters, size, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=1, padding='same',
                             kernel_initializer=initializer, use_bias=False))
  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result

def maxpool():
  result = tf.keras.Sequential()
  result.add(tf.keras.layers.MaxPool2D())

  return result

def deconv2d(filters, size):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=1,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

  result.add(tf.keras.layers.BatchNormalization())
  result.add(tf.keras.layers.ReLU())

  return result

def upsampling2d(filters, size):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

  result.add(tf.keras.layers.BatchNormalization())
  result.add(tf.keras.layers.Dropout(0.4))
  result.add(tf.keras.layers.ReLU())

  return result

def Generator():
  inputs = tf.keras.layers.Input(shape=[512,128,1])
  x = inputs

  # Downsampling through the model
  x = conv2d(32, 3)(x)  # (bs, 512, 128, 32)
  x1 = conv2d(32, 3)(x)  # (bs, 512, 128, 32)
  x = maxpool()(x1)
  x = conv2d(64, 3)(x)  # (bs, 256, 64, 64)
  x2 = conv2d(64, 3)(x)  # (bs, 256, 64, 64)
  x = maxpool()(x2)
  x = conv2d(128, 3)(x)  # (bs, 128, 32, 128)
  x3 = conv2d(128, 3)(x)  # (bs, 128, 32, 128)
  x = maxpool()(x3)
  x = conv2d(256, 3)(x)  # (bs, 64, 16, 256)
  x4 = conv2d(256, 3)(x)  # (bs, 64, 16, 256)
  x = maxpool()(x4)
  x = conv2d(512, 3)(x)  # (bs, 32, 8, 512)
  x = conv2d(512, 3)(x)  # (bs, 32, 8, 512)

  x = upsampling2d(256, 5)(x)  # (bs, 64, 16, 256)
  x = tf.keras.layers.Concatenate()([x, x4])
  x = deconv2d(256, 3)(x)  # (bs, 64, 16, 256)
  x = deconv2d(256, 3)(x)  # (bs, 64, 16, 256)
  x = upsampling2d(128, 5)(x)  # (bs, 128, 32, 128)
  x = tf.keras.layers.Concatenate()([x, x3])
  x = deconv2d(128, 3)(x)  # (bs, 128, 32, 128)
  x = deconv2d(128, 3)(x)  # (bs, 128, 32, 128)
  x = upsampling2d(64, 5)(x)  # (bs, 256, 64, 64)
  x = tf.keras.layers.Concatenate()([x, x2])
  x = deconv2d(64, 3)(x)  # (bs, 256, 64, 64)
  x = deconv2d(64, 3)(x)  # (bs, 256, 64, 64)
  x = upsampling2d(32, 5)(x)  # (bs, 512, 128, 32)
  x = tf.keras.layers.Concatenate()([x, x1])
  x = deconv2d(32, 3)(x)  # (bs, 512, 128, 32)
  x = deconv2d(32, 3)(x)  # (bs, 512, 128, 32)
  x = conv2d(4, 1)(x)  # (bs, 512, 128, 4)

  return tf.keras.Model(inputs=inputs, outputs=x)
