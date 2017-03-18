""" In this example we are going to use tensorboard. Specifically
we are going to visualize the graph and loss of the model. We 
will be using the MNIST data set for our model."""

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST/data", one_hot=True)

""" Lets define our hyper parameters and constants"""
learning_rate = 0.01
training_epoch = 25
batch_size = 100
display_step = 1
logs_path = '/tmp/logs/'

# tf Graph input
# mnist data image with shape of 28*28=784
x = tf.placeholder(tf.float32, [None, 784], name='InputData')
# 0-9 digits recognition => 10 classes
y = tf.placeholder(tf.float32, [None, 10], name='LabelData')

# weights and Bias
W = tf.Variable(tf.zeros([784, 10]), name='Weights')
b = tf.Variable(tf.zeros([10]), name='Bias')

""" In this section we are going to construct the model
and encapsulate all operations into scopes. This is
makes TF Graph visualization more convenient"""

with tf.name_scope('Model'):
	pred = tf.nn.softmax(tf.matmul(x, W) + b)

with tf.name_scope('Loss'):
	cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1)) 

with tf.name_scope('SGD'):
	optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
	
with tf.name_scope('Accuracy'):
	acc = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
	acc = tf.reduce_mean(tf.cast(acc, tf.float32))
	
# initilize the variables
init = tf.global_variables_initializer()

# create a summary to monitor cost tensor
tf.summary.scalar("loss", cost)
# create summary to monitor accuracy tensor
tf.summary.scalar("accuracy", acc)
# merge all sumaries into a single op
merged_summary_op = tf.summary.merge_all()

with tf.Session() as sess:
	sess.run(init)
	
	# op to write logs to TF
	summary_writter = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
	
	# training cycle
	for epoch in range(training_epoch):
		avg_cost = 0.
		total_batch = int(mnist.train.num_examples/batch_size)
		
		# loop over batches
		for i in range(total_batch):
			batch_xs, batch_ys = mnist.train.next_batch(batch_size)
			
			# run optimization op (backprop), cost op
			_, c, summary = sess.run([optimizer, cost, merged_summary_op],
									 feed_dict={x: batch_xs, y: batch_ys})
			# write logs at every iteration
			summary_writter.add_summary(summary, epoch * total_batch + i)
			
			# compute average loss
			avg_cost += c / total_batch
			
		if (epoch+1) % display_step == 0:
			print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost)
			
	print "optimization has finished"
	
	# test model
	print "Accuracy:", acc.eval({x: mnist.test.images, y: mnist.test.labels})

	print "Run the command line:\n" \
		"--> tensorboard --logdir=/tmp/tensorflow_logs " \
		"\nThen open http://0.0.0.0:6006/ into your web browser"
	