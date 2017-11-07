# -*- coding: utf-8 -*-

import tensorflow as tf


class Model:
    def __init__(self,
                 is_train,
                 learning_rate=0.001,
                 learning_rate_decay_factor=0.9995):
        self.x_ = tf.placeholder(tf.float32, [None, 1, 28, 28])
        self.y_ = tf.placeholder(tf.int32, [None])
        self.keep_prob = tf.placeholder(tf.float32)

        x = tf.reshape(self.x_, [-1, 28, 28, 1])

        # TODO: implement input -- Conv -- BN -- ReLU -- MaxPool -- Conv -- BN -- ReLU -- MaxPool -- Linear -- loss
        #        the 10-class prediction output is named as "logits"
        channel_num = 4
        conv1 = tf.contrib.layers.conv2d(x, channel_num, [28, 28], scope='conv_layer1', activation_fn=None)
        bn1 = batch_normalization_layer(conv1, shape=[channel_num])
        relu1 = tf.nn.relu(bn1)
        pool1 = tf.contrib.layers.max_pool2d(relu1, [2, 2], padding='VALID')
        conv2 = tf.contrib.layers.conv2d(pool1, channel_num, [14, 14], scope='conv_layer2', activation_fn=None)
        bn2 = batch_normalization_layer(conv2, shape=[channel_num])
        relu2 = tf.nn.relu(bn2)
        pool2 = tf.contrib.layers.max_pool2d(relu2, [2, 2], padding='VALID')
        # pool2_shape = pool2.get_shape()
        pool2_in_flat = tf.reshape(pool2, [-1, 196])
        logits = tf.contrib.layers.fully_connected(pool2_in_flat, 10, scope='fc_layer1', activation_fn=None)

        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_, logits=logits))
        self.correct_pred = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), self.y_)
        self.pred = tf.argmax(logits, 1)
        self.acc = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

        self.learning_rate = tf.Variable(float(learning_rate), trainable=False,
                                         dtype=tf.float32)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)

        self.global_step = tf.Variable(0, trainable=False)
        self.params = tf.trainable_variables()
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step,
                                                                            var_list=self.params)

        self.saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2,
                                    max_to_keep=3, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)


def weight_variable(shape):  # you can use this func to build new variables
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):  # you can use this func to build new variables
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def batch_normalization_layer(inputs, shape, isTrain=True):
    # TODO: implemented the batch normalization func and applied it on conv and fully-connected layers
    # hint: you can add extra parameters (e.g., shape) if necessary
    all_mean = tf.Variable(tf.constant(0.0, shape=shape), trainable=False)
    all_var = tf.Variable(tf.constant(0.0, shape=shape), trainable=False)
    scale = weight_variable(shape=shape)
    offset = bias_variable(shape=shape)
    const = 0.999
    if isTrain:
        mean, variable = tf.nn.moments(inputs, axes=[0,1,2])
        train_mean = tf.assign(all_mean, all_mean * const + (1 - const) * mean)
        train_var = tf.assign(all_var, all_var * const + (1 - const) * variable)
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs, mean=mean, variance=variable,
                                             offset=offset, scale=scale, variance_epsilon=0.0001)
    else:
        return tf.nn.batch_normalization(inputs, mean=all_mean, variance=all_var,
                                         offset=offset, scale=scale, variance_epsilon=0.0001)

