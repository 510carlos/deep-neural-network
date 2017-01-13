''' Calculate the softmax cross entropy with logits on the MNIST data set '''

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# let's create the model
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

# define the loss optimizer
y_ = tf.placeholder(tf.float32, [None, 10])

# calculate the cross_entropy/cost
cross_entropy = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y) )

# minimize the error with gradient descent
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# let's begin initialization the graph  
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# let's train for 1000 steps
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# lets get the corrected prediction
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
# how accurate is the model
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

