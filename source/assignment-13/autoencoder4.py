# Training one autoencoder at a time by Geron
# need visualization of reconstruction

import tensorflow as tf
import sys
import numpy.random as rnd
from functools import partial
import matplotlib
import matplotlib.pyplot as plt

n_test_digits = 2

def train_autoencoder(X_train, n_neurons, n_epochs, batch_size,
                      learning_rate = 0.01, l2_reg = 0.0005,
                      hidden_activation=tf.nn.elu,
                      output_activation=tf.nn.elu):
    graph = tf.Graph()
    with graph.as_default():
        n_inputs = X_train.shape[1]

        X = tf.placeholder(tf.float32, shape=[None, n_inputs])
        
        my_dense_layer = partial(
            tf.layers.dense,
            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg))

        hidden = my_dense_layer(X, n_neurons, activation=hidden_activation, name="hidden")
        outputs = my_dense_layer(hidden, n_inputs, activation=output_activation, name="outputs")

        reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))

        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss = tf.add_n([reconstruction_loss] + reg_losses)

        optimizer = tf.train.AdamOptimizer(learning_rate)
        training_op = optimizer.minimize(loss)

        init = tf.global_variables_initializer()

    with tf.Session(graph=graph) as sess:
        init.run()
        for epoch in range(n_epochs):
            n_batches = len(X_train) // batch_size
            for iteration in range(n_batches):
                #print("\r{}%".format(100 * iteration // n_batches))
                sys.stdout.flush()
                indices = rnd.permutation(len(X_train))[:batch_size]
                X_batch = X_train[indices]
                sess.run(training_op, feed_dict={X: X_batch})
            loss_train = reconstruction_loss.eval(feed_dict={X: X_batch})
            print("\r{}".format(epoch), "Train MSE:", loss_train)
        params = dict([(var.name, var.eval()) for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)])
        hidden_val = hidden.eval(feed_dict={X: X_train})
        return hidden_val, params["hidden/kernel:0"], params["hidden/bias:0"], params["outputs/kernel:0"], params["outputs/bias:0"]

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/")

hidden_output, W1, b1, W4, b4 = train_autoencoder(mnist.train.images, n_neurons=300, n_epochs=4, batch_size=150,
                                                  output_activation=None)
_, W2, b2, W3, b3 = train_autoencoder(hidden_output, n_neurons=150, n_epochs=4, batch_size=150)

graph = tf.Graph()
with graph.as_default():
    with tf.Session(graph=graph) as sess:
        X_test = mnist.test.images[:n_test_digits]
        activation = tf.nn.elu
        X_place = tf.placeholder(tf.float32, shape=[None, X_test.shape[1]])
        L1 = activation((tf.matmul(X_test, W1) + b1))
        L2 = activation((tf.matmul(L1, W2) + b2))
        L3 = activation((tf.matmul(L2, W3) + b3))
        L4 = tf.matmul(L3, W4) + b4

        def plot_image(image, shape=[28, 28]):
            plt.imshow(image.reshape(shape), cmap="Greys", interpolation="nearest")
            plt.axis("off")


        outputs_val = L4.eval(feed_dict={X_place: X_test})
        fig = plt.figure(figsize=(8, 6))
        for digit_index in range(2):
            plt.subplot(2, 3, digit_index * 3 + 1)
            plot_image(X_test[digit_index])
            plt.subplot(2, 3, digit_index * 3 + 2)
            plot_image(outputs_val[digit_index])

        plt.show()