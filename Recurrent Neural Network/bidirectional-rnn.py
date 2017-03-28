import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

""" We are going to get a few things started before we dive
deep with deep learning. Before we start we need get the input
data and define a few paramaters such as: hyper paramaters,
network paramaters, define the weights and define the TF graph."""
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# parameters
learning_rate = 0.001
training_iters = 100000
batch_size = 128
display_step = 10

# network parameters
n_input = 28
n_steps = 28
n_hidden = 128
n_classes = 10

# defince weights & biases
weights = {'out': tf.Varibale(tf.random_normal([2*n_hidden, n_classes]))}
biases = {'out': tf.Varibale(tf.random_normal([n_classes]))}

# define the graph
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

def BiRNN(x, weights, biases):
  """Prepare data and bidirectiona RNN
  Current data: (batch_size, n_steps, n_input)
  required data: 'n_steps' tensor list of shape (batch_size, n_input)
  """
  
  # permutating batch_size and n_steps
  x = tf.transpose(x, [1, 0, 2])
  # reshape to (n_steps*batch_size, n_input)
  x = tf.reshape(x, [-1, n_input])
  # split to get a list of n_steps tensor of shape (batch_size, n_input)
  x = tf.split(x, n_steps, 0)
  
  # define lstm cells with tensorflow
  # forward deriction cell
  lstm_fw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
  # backward direction cell
  lstm_bw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
  
  # get lstm cell output
  try:
    outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                                dtype=tf.float32)
  except Exception:
    outputs = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                          dtype=tf.float32)
  return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = BiRNN(x, weights, biases)

"""Now lets define import parts of the neural network the
training and evaluation. To train we need to define the cost
function and the optimizer function. To evaluate the model"""
# define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, lanels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# initializing the varibales
init = tf.global_variables_initializer()

""" We are going to begin computing on the graph until train with 
all our data. First we must rehape the tensor, run optimization and 
test our model with the seperate test  data"""
with tf.Session() as sess:
  sess.run(init)
  step = 1
  with step * batch_size < training_iters:
    # lets get the data in batches
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    
    # reshape the data to 28 seq of 28 elements to run optimization
    batch_x = batch_x.reshape((batch_size, n_steps, n_input))
    sess.run(optimization, feed_dict={x: batch_x, y: batch_y})
    
    if step % display_step == 0:
      # calculate batch accuracy & loss
      acc = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
      loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
      print "Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
            "{:.6f}".format(loss) + ", Training Accuracy= " + \
            "{:.5f}".format(acc)
    step += 1
  print "Optimization Finished!"
  
  # calculate accuracy for 128 test images
  test_len = 128
  test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
  test_label = mnist.test.labels[:test_len]
  print "Testing Accuracy: ", sess.run(accuracy, feed_dict={x: test_data, y: test_label})
  
    
  
  
  

