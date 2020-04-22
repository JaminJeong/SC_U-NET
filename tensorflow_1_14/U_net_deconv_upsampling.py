import tensorflow as tf
from tensorflow.keras.layers import concatenate 
from config import *

def U_Net(features, labels, mode) :
    # Input Layer 
    
    input_layer = tf.reshape(features['mag'],[-1,image_width,128,1])
    
    # Convolutional Layer 1 1024 * 128 * 3
    conv1 = tf.layers.batch_normalization(tf.layers.conv2d(inputs = input_layer, filters=32, kernel_size=[3,3], 
                                                       strides=[1,1], padding="same", activation=tf.nn.relu))
    conv1 = tf.layers.batch_normalization(tf.layers.conv2d(inputs = conv1, filters=32, kernel_size=[3,3], 
                                                       strides=[1,1], padding="same", activation=tf.nn.relu))
    max_pooling1 = tf.layers.max_pooling2d(inputs = conv1, pool_size = 2, strides = 2)
    
    # Convolutional Layer 2 512 * 64 * 32
    conv2 = tf.layers.batch_normalization(tf.layers.conv2d(inputs = max_pooling1, filters = 64, kernel_size = [3,3], 
                                                       strides = [1,1], padding="same", activation = tf.nn.relu))
    conv2 = tf.layers.batch_normalization(tf.layers.conv2d(inputs = conv2, filters = 64, kernel_size = [3,3], 
                                                       strides = [1,1], padding="same", activation = tf.nn.relu))
    max_pooling2 = tf.layers.max_pooling2d(inputs = conv2, pool_size = 2, strides = 2)

    # Convolutional Layer 3 256 * 32 * 64
    conv3 = tf.layers.batch_normalization(tf.layers.conv2d(inputs = max_pooling2, filters = 128, kernel_size = [3,3], 
                                                       strides = [1,1], padding="same", activation = tf.nn.relu))
    conv3 = tf.layers.batch_normalization(tf.layers.conv2d(inputs = conv3, filters = 128, kernel_size = [3,3], 
                                                       strides = [1,1], padding="same", activation = tf.nn.relu))
    max_pooling3 = tf.layers.max_pooling2d(inputs = conv3, pool_size = 2, strides = 2)

    # Convolutional Layer 4 128 * 16 * 128
    conv4 = tf.layers.batch_normalization(tf.layers.conv2d(inputs = max_pooling3, filters = 256, kernel_size = [3,3], 
                                                       strides = [1,1], padding="same", activation = tf.nn.relu))
    conv4 = tf.layers.batch_normalization(tf.layers.conv2d(inputs = conv4, filters = 256, kernel_size = [3,3], 
                                                       strides = [1,1], padding="same", activation = tf.nn.relu))
    max_pooling4 = tf.layers.max_pooling2d(inputs = conv4, pool_size = 2, strides = 2)

    # Convolutional Layer 5 64 * 8 * 256
    conv5 = tf.layers.batch_normalization(tf.layers.conv2d(inputs = max_pooling4, filters = 512, kernel_size = [3,3], 
                                                       strides = [1,1], padding="same", activation = tf.nn.relu))
    conv5 = tf.layers.batch_normalization(tf.layers.conv2d(inputs = conv5, filters = 512, kernel_size = [3,3], 
                                                       strides = [1,1], padding="same", activation = tf.nn.relu))

    # Deconvolutional Layer1 (upsampling)
    deconv1 = tf.image.resize_images(images = conv5, size = (conv5.shape[1]*2, conv5.shape[2]*2), method=tf.image.ResizeMethod.BILINEAR)
    deconv1 = tf.layers.batch_normalization(tf.layers.conv2d(inputs = deconv1, filters = 256, kernel_size = [1,1],
                                                             strides = [1,1], padding="same", activation = tf.nn.relu))

    # Deconvolutional Layer2 (deconv) 128 * 16 * 512
    deconv2 = tf.layers.batch_normalization(tf.layers.conv2d(inputs = concatenate([deconv1,conv4],3), filters = 256, kernel_size = [3,3],
                                                                 strides = [1,1], padding="same", activation = tf.nn.relu))
    deconv2 = tf.layers.batch_normalization(tf.layers.conv2d(inputs = deconv2, filters = 256, kernel_size = [3,3],
                                                                 strides = [1,1], padding="same", activation = tf.nn.relu))

    # Deconvolutional Layer3 (upsampling)
    deconv3 = tf.image.resize_images(images = deconv2, size = (deconv2.shape[1]*2, deconv2.shape[2]*2), method = tf.image.ResizeMethod.BILINEAR)
    deconv3 = tf.layers.batch_normalization(tf.layers.conv2d(inputs = deconv3, filters = 128, kernel_size = [1,1],
                                                             strides = [1,1], padding="same", activation = tf.nn.relu))
    # Deconvolutional Layer4 (deconv)
    deconv4 = tf.layers.batch_normalization(tf.layers.conv2d(inputs = concatenate([deconv3,conv3],3), filters = 128, kernel_size = [3,3],
                                                                 strides = [1,1], padding="same", activation = tf.nn.relu))
    deconv4 = tf.layers.batch_normalization(tf.layers.conv2d(inputs = deconv4, filters = 128, kernel_size = [3,3],
                                                                 strides = [1,1], padding="same", activation = tf.nn.relu))

    # Deconvolutional Layer5 (upsampling)
    deconv5 = tf.image.resize_images(images = deconv4, size = (deconv4.shape[1]*2, deconv4.shape[2]*2), method = tf.image.ResizeMethod.BILINEAR)
    deconv5 = tf.layers.batch_normalization(tf.layers.conv2d(inputs = deconv5, filters = 64, kernel_size = [1,1],
                                                             strides = [1,1], padding="same", activation = tf.nn.relu))
    # Deconvolutional Layer6 (deconv)
    deconv6 = tf.layers.batch_normalization(tf.layers.conv2d(inputs = concatenate([deconv5,conv2],3), filters = 64, kernel_size = [3,3],
                                                                 strides = [1,1], padding="same", activation = tf.nn.relu))
    deconv6 = tf.layers.batch_normalization(tf.layers.conv2d(inputs = deconv6, filters = 64, kernel_size = [3,3],
                                                                 strides = [1,1], padding="same", activation = tf.nn.relu))

    # Deconvolutional Layer7 (upsampling)
    deconv7 = tf.image.resize_images(images = deconv6, size = (deconv6.shape[1]*2, deconv6.shape[2]*2), method = tf.image.ResizeMethod.BILINEAR)
    deconv7 = tf.layers.batch_normalization(tf.layers.conv2d(inputs = deconv7, filters = 32, kernel_size = [1,1],
                                                             strides = [1,1], padding="same", activation = tf.nn.relu))
    # Deconvolutional Layer8 (deconv)
    deconv8 = tf.layers.batch_normalization(tf.layers.conv2d(inputs = concatenate([deconv7,conv1],3), filters = 32, kernel_size = [3,3], 
                                                                 strides = [1,1], padding="same", activation = tf.nn.relu))
    deconv8 = tf.layers.batch_normalization(tf.layers.conv2d(inputs = deconv8, filters = 32, kernel_size = [3,3], 
                                                                 strides = [1,1], padding="same", activation = tf.nn.relu))

    # Final Convolutional Layer
    deconv9 = tf.layers.batch_normalization(tf.layers.conv2d(inputs = deconv8, filters = 4, kernel_size = [1,1],
                                                       strides = [1,1], padding="same", activation = tf.nn.relu))
 
 
    predictions = {'outputs': deconv9
                  }
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    loss = tf.losses.absolute_difference(labels,deconv9)
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss)
