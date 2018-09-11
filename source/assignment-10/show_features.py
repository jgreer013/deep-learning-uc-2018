import tensorflow as tf 

saver = tf.train.import_meta_graph("./my_mnist_model.meta")

ker = tf.get_default_graph().get_tensor_by_name("conv1/kernel/Adam:0")

print ker.get_shape()

sess = tf.Session()

saver.restore(sess, tf.train.latest_checkpoint("./"))

features1 = sess.run(ker)

print features1[:,:,0,0]

print features1[:,:,0,1]

print features1[:,:,0,2]