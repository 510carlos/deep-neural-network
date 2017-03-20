"""In this example we are going to go over on how to save and restore a model
with tensorflow. We are going to use the MNIST dataset. """

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Hyper Parameters
learning_rate = 0.001
batch_size = 100
display_step = 1
model_path = "/tmp/model.ckpt"

# network parameters
n_hidden_1 = 256
n_hidden_2 = 256
n_input = 784
n_classes = 10

# lets define the graph
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

# lets create the models
def multilayer_perceptron(x, weights, biases):
	# 1st layer with relua activation
	layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
	layer_1 = tf.nn.relu(layer_1)
	
	# 2nd layer
	layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
	layer_2 = tf.nn.relu(layer_2)
	
	# output layee
	out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
	return out_layer
	
weights = {
	'h1' : tf.Variable(tf.random_normal([n_input, n_hidden_1])),
	'h2' : tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
	'out' : tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
	'b1' : tf.Variable(tf.random_normal([n_hidden_1])),
	'b2' : tf.Variable(tf.random_normal([n_hidden_2])),
	'out' : tf.Variable(tf.random_normal([n_classes])),
}

# contruct model
pred = multilayer_perceptron(x, weights, biases)

# define lost + optimizer
cost = tf.reduce_mean(
		tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
		
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# initialize the variabkes
init = tf.global_variables_initializer()

# saver 
saver = tf.train.Saver()

print "Lets start the first session..."
with tf.Session() as sess:
	# initilize variables
	sess.run(init)
	
	# training cycle
	for epoch in range(3):
		avg_cost = 0
		total_batch = int(mnist.train.num_examples/batch_size)
		
		for i in range(total_batch):
			batch_x, batch_y = mnist.train.next_batch(batch_size)
			
			_, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
														  y: batch_y})
			
			# compute the average loss
			avg_cost += c / total_batch
		# display logs per epoch step
		if epoch % display_step == 0:
			print "Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost)
	print "First optimization finished!"
	
	# test model
	correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
	
	# calculate accuracy
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	
	print "Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}) 
	
	# lets save the model to disk
	save_path = saver.save(sess, model_path)
	
	print "Model saved in file: %s" % save_path

print "Starting the 2nd session"

with tf.Session() as sess:
	# initilize variables
	sess.run(init)
	
	# restore model weights
	load_path = saver.restore(sess, model_path)
	
	# training cycle
	for epoch in range(3):
		avg_cost = 0
		total_batch = int(mnist.train.num_examples/batch_size)
		
		for i in range(total_batch):
			batch_x, batch_y = mnist.train.next_batch(batch_size)
			
			_, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
														  y: batch_y})
			
			# compute the average loss
			avg_cost += c / total_batch
		# display logs per epoch step
		if epoch % display_step == 0:
			print "Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost)
	print "First optimization finished!"
	
	# test model
	correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
	
	# calculate accuracy
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	
	print "Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}) 