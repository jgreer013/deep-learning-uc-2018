import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

data = fetch_california_housing()
scaler = StandardScaler()
scaler.fit(data.data)
norm_data = scaler.transform(data.data)

m, n = data.data.shape
data_b = np.c_[np.ones((m, 1)), norm_data]

X = tf.constant(data_b, dtype=tf.float32, name="X")
y = tf.constant(data.target.reshape(-1, 1), dtype=tf.float32, name="y")
XT = tf.transpose(X)

theta = tf.Variable(tf.random_uniform([n+1, 1], -1.0, 1.0), name="theta")

y_pred = tf.matmul(X, theta, name="predictions")

error = y_pred - y

mse = tf.reduce_mean(tf.square(error), name="mse")

# Manual Diff
#print m
#gradients = tf.matmul(XT, error)
# Note: 2/m was not multiplied due to diminishing gradients
# AutoDiff
gradients = tf.gradients(mse, [theta])[0]

#learning_rate = 0.00004 # largest learning rate that converges for manual diff
learning_rate = 0.01

training_op = tf.assign(theta, theta - learning_rate * gradients)

init = tf.global_variables_initializer()

n_epochs = 1000

with tf.Session() as sess:
    sess.run(init)
    for i in xrange(n_epochs):
        print(mse.eval())
        #print gradients.eval()
        sess.run(training_op)

        if i == 999:
            g = tf.get_default_graph()
            op = g.get_operations()
            writer = tf.summary.FileWriter('logs', sess.graph)
            print sess.run(training_op)
            writer.close()
            print op