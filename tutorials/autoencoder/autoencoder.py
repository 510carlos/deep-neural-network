from __future__ import division, print_function, absolute_import

""" In this example we are going to to use the MNIST data set
and apply an autoencoder. What that means we are going to encoder
an image and then decode so we can compare how accurately the
model can re-create the image. At the end we are going to compare
the original image with the encoder image."""

""" We are going to load all the dependancies. Then we are going
to load the MNIST dataset so we can work with it."""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# import mnist
from tensorflow.examples.tutorials.mnist import imput_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

"""First lets start by defining the hyper paramaters such as:
learning rate, training epochs, batch size, display step
and examples to show."""
# paramaters
learning_rate = 0.01
training_epochs = 20
batch_size = 256
display_step = 1
examples_to_show = 10

"""Now we are going to begin to define the network and
create the tensor represantations of the neural network."""
# network paramaters
n_hidden_1 = 256 # 1st later
n_hidden_2 = 128 # 2nd layer
n_input = 784 # (28*28) input

X = tf.placeholder("float", [None, n_input])

# we are going to define a 2 layer encoder/decoder to store the weights
# and we can't forget the biases
weights = {
  'encoder_h1' : tf.Variable(tf.random_normal([n_input, n_hidden_1])),
  'encoder_h2' : tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
  'decoder_h1' : tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
  'decoder_h1' : tf.Variable(tf.random_normal([n_hidden_1, n_input])),
}
biases = {
  'encoder_b1' : tf.Variable(tf.random_normal([n_hidden_1])),
  'encoder_b2' : tf.Variable(tf.random_normal([n_hidden_2])),
  'decoder_b1' : tf.Variable(tf.random_normal([n_hidden_1])),
  'decoder_b2' : tf.Variable(tf.random_normal([n_input])),
}

"""Now we need to define both the encoder and the decoder functions.
Each will hold 2 layers and activatited with a sigmoid function."""
def encoder(x):
  # lets test this
  #layer_1 = tf.nn.sigmoid(tf.matmul(x, weights['encoder_h1']) + biases['encoder_b1'])
  # encoder hidden layer with sigmoid activation #1
  layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                biases['encoder_b1']))
  # decoder hidden layer with sigmoid #2
  layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                biases['encoder_b2']))
  
  return layer_2

def decoder(x):
  """Building the decoder"""
  # encoder hidden layer w/ sigmoid
  layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                biases['decoder_b2']))
  # decoder hidden layer w/ sigmoid
  layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_b2']),
                                biases['decoder_b2']))
  return layer_2

"""Now that we have the encoder and decoder defined we can pass the
the image to the encoder then pass the encoder to the decoder. We
are going to save the predicted result and save the actual result.
Next, we need to defice our cost and optimizer functions. We will
be reducing the mean for the cost function and using the """
# construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# prediction
y_pred = decoder_op
# targets (labels) are the input data
y_true = X

# define loss and optimizer, minimize the squared error
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize()

"""Finally, we can start training with the tensorflow graph. We
are going to iterate over the training_epochs and the total_batches.
to run the optimizer and cost function operations."""
# initilize the variables
init = tf.global_variables_initializer()

# launch the graph
sess = tf.InteractiveSession()
sess.run(init)

total_batch = int(mnist.train.num_examples / batch_size)
# training cycle
for epoch in range(training_epochs):
  # loop over all batches
  for i in range(total_batch):
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    # run optimization and cost
    _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
  # display logs per epoch step
  if epoch % display_step == 0:
    print("Epoch:", '%04d' % (epoch+1),
          "cost=", "{:.9f}".format(c))

print "Opt has completed"

"""We are going to test our model with a test data set. After
we compute the the encoder and decoder we are going show both
original image and decoded image side by side."""
# Applying encode and decode over test set
encode_decode = sess.run(
  y_pred, feed_dict={X: mnist.test.images[:examples_to_show]})
# compare original images with their reconstructions
f, a = plt.subplots(2, 10, figsize={10, 2})
for i in range(examples_to_show):
  a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
  a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
f.show()
plt.draw()


  
  
  
  