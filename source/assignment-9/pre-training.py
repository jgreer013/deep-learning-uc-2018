from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import time
from matplotlib import pyplot as plt

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

he_init = tf.contrib.layers.variance_scaling_initializer()

learning_rate = 0.01
momentum = 0.95

def dnn(inputs, n_hidden_layers=5, n_neurons=100, name=None,
        activation=tf.nn.elu, initializer=he_init):
    with tf.variable_scope(name, "dnn"):
        for layer in range(n_hidden_layers):
            inputs = tf.layers.dense(inputs, n_neurons, activation=activation, 
                                     kernel_initializer=initializer, 
                                     name="hidden%d" % (layer+1))
        return inputs 

## 10.1
mnist = input_data.read_data_sets("/tmp/data/")

X_train_1 = mnist.train.images
y_train_1 = mnist.train.labels

X_train_2 = mnist.validation.images
y_train_2 = mnist.validation.labels

X_test = mnist.test.images
y_test = mnist.test.labels


n_inputs = 28*28
reset_graph()

X = tf.placeholder(tf.float32, shape=(None, 2, n_inputs), name="X")
X1, X2 = tf.unstack(X, axis=1)

y = tf.placeholder(tf.int32, shape=[None, 1])

dnn1 = dnn(X1, name="DNN_A")
dnn2 = dnn(X2, name="DNN_B")

dnn_outputs = tf.concat([dnn1, dnn2], axis=1)

hidden = tf.layers.dense(dnn_outputs, units=10, activation=tf.nn.elu, kernel_initializer=he_init)
logits = tf.layers.dense(hidden, units=1, kernel_initializer=he_init)
y_proba = tf.nn.sigmoid(logits)

y_pred = tf.cast(tf.greater_equal(logits, 0), tf.int32)

y_as_float = tf.cast(y, tf.float32)
xentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_as_float, logits=logits)
loss = tf.reduce_mean(xentropy)



optimizer = tf.train.MomentumOptimizer(learning_rate, momentum, use_nesterov=True)
training_op = optimizer.minimize(loss)

y_pred_correct = tf.equal(y_pred, y)
accuracy = tf.reduce_mean(tf.cast(y_pred_correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

## 10.2


def make_batch(images, labels, batch_size):
    size_same = batch_size // 2

    x = []
    y = []
    while len(x) < size_same:
        index1, index2 = np.random.randint(0, len(images), 2)
        if index1 != index2 and labels[index1] == labels[index2]:
            x.append(np.array([images[index1], images[index2]]))
            y.append([1])
    
    while len(x) < batch_size:
        index1, index2 = np.random.randint(0, len(images), 2)
        if index1 != index2 and labels[index1] != labels[index2]:
            x.append(np.array([images[index1], images[index2]]))
            y.append([0])

    indices = np.random.permutation(batch_size)
    return np.array(x)[indices], np.array(y)[indices]

## 10.3

X_test1, y_test1 = make_batch(X_test, y_test, len(X_test))

n_epochs = 100
batch_size = 512

with tf.Session() as sess:
    init.run()
    for epoch in xrange(n_epochs):
        for iteration in xrange(mnist.train.num_examples // batch_size):
            X_batch, y_batch = make_batch(X_train_1, y_train_1, batch_size)
            loss_val, _ = sess.run([loss, training_op], feed_dict={X:X_batch, y:y_batch})
        print(epoch, "Train loss:", loss_val)
        if epoch % 5 == 0:
            acc_test = accuracy.eval(feed_dict={X:X_test1, y:y_test1})
            print(epoch, "Test Accuracy:", acc_test)
        
    save_path = saver.save(sess, "./my_digit_comparison_model.ckpt")

## 10.4
reset_graph()

n_inputs = 28*28
n_outputs = 10

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int32, shape=(None), name="y")

dnn_outputs = dnn(X, name="DNN_A")
frozen_outputs = tf.stop_gradient(dnn_outputs)

logits = tf.layers.dense(dnn_outputs, n_outputs, kernel_initializer=he_init)
y_proba = tf.nn.softmax(logits)

xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
loss = tf.reduce_mean(xentropy, name="loss")

optimizer = tf.train.MomentumOptimizer(learning_rate, momentum, use_nesterov=True)
training_op = optimizer.minimize(loss)

correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()

dnn_A_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="DNN_A")
restore_saver = tf.train.Saver(var_list={var.op.name: var for var in dnn_A_vars})
saver = tf.train.Saver()

n_epochs = 100
batch_size = 64

with tf.Session() as sess:
    init.run()
    restore_saver.restore(sess, "./my_digit_comparison_model.ckpt")

    for epoch in xrange(n_epochs):
        index = np.random.permutation(len(X_train_2))
        for indices in np.array_split(index, len(X_train_2) // batch_size):
            X_batch, y_batch = X_train_2[indices], y_train_2[indices]
            sess.run(training_op, feed_dict={X: X_batch, y:y_batch})
        if epoch % 10 == 0:
            acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
            print(epoch, "Test accuracy:", acc_test)

    save_path = saver.save(sess, "./my_mnist_model_final.ckpt")
