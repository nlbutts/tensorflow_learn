import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib.layers import batch_norm
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.preprocessing import StandardScaler
import numpy as np

n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name='y')

is_training = tf.placeholder(tf.bool, shape=(), name="is_training")
bn_params = {
    'is_training': is_training,
    'decay': 0.99,
    'updates_collections': None
}

with tf.name_scope("dnn"):
    hidden1 = fully_connected(X,        n_hidden1, scope="hidden1", normalizer_fn=batch_norm, normalizer_params=bn_params)
    hidden2 = fully_connected(hidden1,  n_hidden2, scope="hidden2", normalizer_fn=batch_norm, normalizer_params=bn_params)
    logits  = fully_connected(hidden2,  n_outputs, scope="outputs", activation_fn=None, normalizer_fn=batch_norm, normalizer_params=bn_params)

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

    learning_rate = 0.01

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

mnist = input_data.read_data_sets("./data")

n_epochs = 400
batch_size = 50

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={is_training: True, X: X_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict={is_training: False, X: X_batch, y: y_batch})
        acc_test  = accuracy.eval(feed_dict={is_training: False, X: mnist.test.images, y: mnist.test.labels})
        print(epoch, "Train accuracy: ", acc_train, "Test accuracy: ", acc_test)
        save_path = saver.save(sess, "./model/model")