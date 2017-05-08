""" In this example we are going to create a 
Dynamic Recurrent Netowrk (LSTM). A dynamic RNN are
particuly useful when we are accept a max seq length.
The input can be any length seq but will have to be 
padded with zeros to keep the np array cosistent."""

import tensorflow as tf
import random

"""First we need to generate some data so we can start."""
# toy random data generator
class ToySequenceData(object):
  """
    Generate squence of data with dynamic length.
    Geenrates samples for training:
    - Class 0: linear seq [0, 1, 2, 3, 4...]
    - Class 1: Random Seq [1, 3, 10, ,7...]
    
    Notice: we must pad each seq with zeros
    to the max seq len to keep the consistency.
  """
  def __init__(self, n_samples=1000, max_seq_len=20, min_seq_len=3,
              max_value=1000):
    self.data = []
    self.labels = []
    self.seqlen = []
    for i in range(n_samples):
      # random seq length
      len = random.randint(min_seq_len, max_seq_len)
      # monitor sequence length for TF dynamic calculation
      self.seqlen.append(len)
      # monitor sequence length int sequence (50% prob)
      if random.random() < .5:
        # generate a linear sequence
        rand_start = random.randint(0, max_value - len)
        s = [[float(i)/max_value] for i in
            range(rand_start, rand_start + len)]
        # pad seq for dimensions consistency
        s += [[0.] for i in range(max_seq_len - len)]
        self.data.append(s)
        self.labels.append([1., 0.])
      else:
        # generat a ramdom seq
        s = [[float(random.randint(0, max_value))/max_value]
            for i in range(len)]
        # pad seq for dimensions consistency
        s += [[0.] for i in range(max_seq_len - len)]
        self.data.append(s)
        self.labels.append([0., 1.])
    self.batch_id = 0
    
  def next(self, batch_size):
    """ 
      Return a batch of data. when dataser end is readched,
      start over.
    """
    if self.batch_id == len(self.data):
      self.batch_id = 0
    batch_data = (self.data[self.batch_id:min(self.batch_id +
                                             batch_size, len(self.data))])
    batch_labels = (self.labels[self.batch_id:min(self.batch_id +
                                                 batch_size, len(self.data))])
    batch_seqlen = (self.seqlen[self.batch_id:min(self.batch_id +
                                                 batch_size, len(self.data))])
    self.batch_id = min(self.batch_id + batch_size, len(self.data))
    return batch_data, batch_labels, batch_seqlen

"""Now that we have data we can begin building the model. We
are going to start with the hyper parameters, network parameters,
get some data, define the TF graph, and define weights + biases."""
# model

# hyper paramaters
learning_rate = 0.01
training_iters = 100000
batch_size = 10
display_step = 10

# network parameters
seq_max_len = 20
n_hidden = 64
n_classes = 2

trainset = ToySequenceData(n_samples=1000, max_seq_len=seq_max_len)
testset = ToySequenceData(n_samples=500, max_seq_len=seq_max_len)

# The TF graph
x = tf.placeholder("float", [None, seq_max_len, 1])
y = tf.placeholder("float", [None, n_classes])
seqlen = tf.placeholder(tf.int32, [None])

# the weights

weights = {'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))}
biases = {'out': tf.Variable(tf.random_normal([n_classes]))}

"""Next step is to design the dynamic RNN. We are going to reshape 
the data to match the RNN requirements. After we reshape we are going to
define the lstm cell. Now that we have the lstm we can perform the
dynamic calculation. When we perform a dynamic calculation we need to 
retrieve the last output. To retain the last ouputs we must 
build an index system. Next we pass the data through an activation op."""
def dynamicRNN(x, seqlen, weights, biases):
  
    # prepare data shape to match 'rnn' function requirements
    # current data input shape: (batch_size, n_steps, n_input)
    # required shape: n_steps tensor list of shape (batch_size, n_input)

    # permutating batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
    # reshaping to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, 1])
    # split to get a list of n_steps tensor of shape (batch_size, n_input)
    x = tf.split(x, seq_max_len, 0)

    # define a lstm cell with TF
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)

    # get lstm cell output, providing sequence_length withh perform
    # dynamic calculation
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32,
                                sequence_length=seqlen)
    # when performing dynamic calculation, we must retrieve the last
    # dynamic computer output. If a sequence length is 10,
    # we need to retrieve the 10 output

    # However TF doesn't support advanced indexing yet, so we build
    # ac sutom op that for each sample in batch size, get its length
    # and get the corresponding relevant output

    # outputs is a lost of putput at every timestep, we pack
    # them in a tensor and change back dimensions to
    # [batch_size, n_steps, n_input]
    outputs = tf.stack(outputs)
    outputs = tf.transpose(outputs, [1, 0, 2])

    # hack to build the index and retrieval
    batch_size = tf.shape(outputs)[0]
    # start index for each sample
    index = tf.range(0, batch_size) * seq_max_len + (seqlen - 1)
    # indexing
    outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index)

    # linear activation using outputs computed above
    return tf.matmul(outputs, weights['out']) + biases['out']

""" Now that we have defined the dynamic RNN we can initialize it.
After, we initialize we will define the cost function and the
optimizer function. Next, we can define how we want to eveluate
model by optaining the cost corred_pred and accuracy."""
pred = dynamicRNN(x, seqlen, weights, biases)

# define lost and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# evaluate the model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

"""We can finally compute the graph. """
# initialize the variables
init = tf.global_variables_initializer()

#launch graph
with tf.Session() as sess:
  sess.run(init)
  step = 1
  while step * batch_size < training_iters:
    batch_x, batch_y, batch_seqlen = trainset.next(batch_size)
    # run optimizer op (backprop)
    sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                       seqlen: batch_seqlen})
    if step % display_step == 0:
      # calc accuracy
      acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y,
                                        seqlen: batch_seqlen})
      loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y,
                                        seqlen: batch_seqlen})
      print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
            "{:.6f}".format(loss) + ", Training Accuracy= " + \
            "{:.5f}".format(acc))
    step += 1
  print "Opt complete"
  
  # calculate accuracy
  test_data = testset.data
  test_label = testset.labels
  test_seqlen = testset.seqlen
  print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: test_data, y: test_label,
                                      seqlen: test_seqlen}))