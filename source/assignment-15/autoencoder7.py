# denoising autoencoder of Geron, using dropout
import tensorflow as tf

n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 150  # codings
n_hidden3 = n_hidden1
n_outputs = n_inputs

learning_rate = 0.01

dropout_rate = 0.3
noise_level = dropout_rate

training = tf.placeholder_with_default(False, shape=(), name='training')

X = tf.placeholder(tf.float32, shape=[None, n_inputs])
X_drop = tf.layers.dropout(X, dropout_rate, training=training)

hidden1 = tf.layers.dense(X_drop, n_hidden1, activation=tf.nn.relu,
                          name="hidden1")
hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu, # not shown in the book
                          name="hidden2")                            # not shown
hidden3 = tf.layers.dense(hidden2, n_hidden3, activation=tf.nn.relu, # not shown
                          name="hidden3")                            # not shown
outputs = tf.layers.dense(hidden3, n_outputs, name="outputs")        # not shown

reconstruction_loss = tf.reduce_mean(tf.square(outputs - X)) # MSE

optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(reconstruction_loss)
    
init = tf.global_variables_initializer()

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/")

n_epochs = 10
batch_size = 150

def right_half(x, size):
    if (x - size*(x / size)) >= size/2:
        return True
    else:
        return False

right_side_range = [x for x in range(28*28) if right_half(x, 28)]


import sys
X_test = mnist.test.images[:2]
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        n_batches = mnist.train.num_examples // batch_size
        for iteration in range(n_batches):
            print("\r{}%".format(100 * iteration // n_batches))
            sys.stdout.flush()
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch, training: True})
        loss_train = reconstruction_loss.eval(feed_dict={X: X_batch})
        print("\r{}".format(epoch), "Train MSE:", loss_train)
    state = tf.random_normal(shape=tf.shape(X_test)).eval()
    state = outputs.eval(feed_dict={X: state})
    state = state + noise_level * tf.random_normal(tf.shape(X_test))
    state = outputs.eval(feed_dict = {X: state.eval()})
    state[:,right_side_range] = X_test[:, right_side_range]
    outputs_val = outputs.eval(feed_dict={X:state})


import matplotlib
import matplotlib.pyplot as plt

def plot_image(image, shape=[28, 28]):
    plt.imshow(image.reshape(shape), cmap="Greys", interpolation="nearest")
    plt.axis("off")

fig = plt.figure(figsize=(8, 6))
for digit_index in range(2):
    plt.subplot(2, 3, digit_index * 3 + 1)
    plot_image(X_test[digit_index])
    plt.subplot(2, 3, digit_index * 3 + 2)
    plot_image(outputs_val[digit_index])

plt.show()