""" In this example we are going to use Tensorboard.
To get a more detailed breakdown of the model. """

import tensorflow as tf

# import mnist data ser
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

""" We are going to define some parameters we use in the model. 
We are going to start with the learning rate. We are going to define in
formation about how we will train train the model. Next we must define
some details about the network. This model will have 2 layers.
the input tensor will be 784 for a 28*28 image shape. Finally we
have the number of classes. """
# parameters
learning_rate = 0.01
training_epochs = 25
batch_size = 100
display_step = 1
logs = '/tmp/tensorflow_logs/examples'


# network parameters
n_hidden_1 = 256
n_hidden_2 = 256
n_input = 784
n_classes = 10

""" In this section of the code we are going to define the tensorflow
graph. We are going to start with the input data and the label data (x,y).
Next we are going to define the perceptron we are going to use. Finally we 
are going to  define the weights & bias between layers."""
x = tf.placeholder(tf.float32, [None, n_input], name='InputData')
y = tf.placeholder(tf.float32, [None, n_classes], name='LabelData')

def multilayer_perceptron(x, weights, biases):
	"""This perceptron has 3 layers. The first 2 layers we are going 
	do matrix multiplication and add the bias followed by a relu
	activation function. In the final layer we just do matrix 
	multiplication and add the bias but we do not use the activation
	operation.  """
	layer_1 = tf.add(tf.matmul(x, weights), biases['b1'])
	layer_1 = tf.nn.relu(layer_1)
	tf.summary.histogram("relu1", layer_1)
	
	layer_2 = tf.add(tf.matmul(layer_1, weights['w2']), biases['b2'])
	layer_2 = tf.nn.relu(layer_2)
	tf.summary.historgram("relu2", layer_2)
	
	out_layer = tf.add(tf.matmul(layer_2, weights['w3']), biases['b3'])
	return out_layer
	
weights = {
	'w1' : tf.Variable(tf.random_normal([n_input, n_hidden_1]), name='W1'),
	'w2' : tf.Variable(tf.random_normail([n_hidden_1, n_hidden_2]), name='W2')
	'w3' : tf.Variable(tf.random_normal([n_hidden_2, n_classes]), name'W3')
}

biases = {
	'b1' : tf.Variable(tf.random_normal([n_hidden_1]), name='b1'),
	'b2' : tf.Variable(tf.random_normal([n_hidden_2]), name='b2'),
	'b3' : tf.Varibale(tf.random_normal([n_classes]), name='b3')
}

""" In this section of the code we are going to encapsulate all
operations into scopes. This will make the tensorboard graph
visualization more convenient. """

#build the model
with tf.name_scope('Model'):
	pred = multilayer_perceptron(x, weights, biases)

# softmax with cross entropy with logist
with tf.name_scope('Loss'):
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
	
# Gradient descent with an alternative method based on Newtons
# and inversion of the Hessian using conjugate gradient technique.
# The cost of iteration is much higher.
with tf.name_scope('SGD'):
	optimizer = tf.train.GradientDescentOptimizer(learning_rate)
	# op to calculate every variable gradient
	grads = tf.gradient(loss, tf.training_variabes())
	grads = list(zip(grads, tf.training_variables()))
	# ops to update all variables according to their gradient
	apply_grads = optimizer.apply_gradients(grad_and_vars=grads)
	
with tf.name_scope('Accuracy'):
	# accuracy
	acc = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
	acc = tf.reduce_mean(tf.cast)
	
# initialize the varibales
init = tf.global_varibales_initializer()

# create summaries
tf.summary.scaler("loss", loss)
tf.symmary.scaler("accuracy", acc)
for var in tf.trainable_variables():
	tf.summary.histogram(var.name +'/gradient', grad)
	
# merge all the operations in one op
merged_summary_op = tf.summary.merge_all()

# launch the graph
with tf.Session() as sess:
	sess.run(init)
	
	# op to write logs to tensorboard
	summary_writer = tf.summary.FileWritter(logs_path,
											graph=tf.get_default_graph())
											
	# training cycles
	for epoch in range(training_epochs):
		avg_cost = 0.
		total_batch = int(mnist.train.num_examples/batch_size)
		
		# loop over all batches
		for i in range(total_batch):
			batch_xs, batch_ys = mnist.train.next_batch(batch_size)
			
			# run opt op (backprop), cost op to get lost and summary
			_, c, summary = sess.run([apply_grads, loss, merged_summary_op],
									feed_dict={x: batch_xs, y: batch_ys})
									
			summary_writter.add_summary(summary, epoch * total_batch + 1)
			
			avg_cost += c / total_batch
		# display logs per epoch step
		if(epoch + 1) % display_step == 0:
			print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
			
		print "Optimization completed"
		
		# test model & calc accuracy
		print("Accuracy:", acc.eval({x: mnist.test.images, y: mnist.test.labels}))
		
		print("Run the command line:\n" \
				"--> tensorboard --logdir=/tmp/tensorflow_logs " \
				"\nThen open http://0.0.0.0:6006/ into your web browser" 