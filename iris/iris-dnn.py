from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

# Data Sets
IRIS_TRAINING = "iris_training.csv"
IRIS_TEST = "iris_test.csv"

#load datasets
training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
	filename=IRIS_TRAINING,
	target_dtype=np.int,
	features_dtype=np.float32)
	
test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
	filename=IRIS_TEST,
	target_dtype=np.int,
	features_dtype=np.float32)
	
# ## Construct a deep Neural network classifier

# Specify that all features have real-value data
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]

# build 3 layer DNN with 10, 20, 10 units
classifier = tf.contrib.learn.DNNClassifier(
	feature_columns=feature_columns,
	hidden_units=[10, 20, 10],
	n_classes=3,
	model_dir="/tmp/iris_model")
	
# ## Fit the DNNClassifier to the Iris Training Data
# fit model
classifier.fit(x=training_set.data, y=training_set.target, steps=2000)

# ## Evaluate Model Accuracy
accuracy_score = classifier.evaluate(x=test_set.data, y=test_set.target)["accuracy"]

print('Accuracy: {0:f}'.format(accuracy_score))











































'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

# Data sets
IRIS_TRAINING = "iris_training.csv"
IRIS_TEST = "iris_test.csv"

# Load datasets.
training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=IRIS_TRAINING,
    target_dtype=np.int,
    features_dtype=np.float32)
test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=IRIS_TEST,
    target_dtype=np.int,
    features_dtype=np.float32)

# Specify that all features have real-value data
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]

# Build 3 layer DNN with 10, 20, 10 units respectively.
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10, 20, 10],
                                            n_classes=3,
                                            model_dir="/tmp/iris_model")

# Fit model.
classifier.fit(x=training_set.data,
               y=training_set.target,
               steps=2000)

# Evaluate accuracy.
accuracy_score = classifier.evaluate(x=test_set.data,
                                     y=test_set.target)["accuracy"]
print('Accuracy: {0:f}'.format(accuracy_score))

# Classify two new flower samples.
new_samples = np.array(
    [[6.4, 3.2, 4.5, 1.5], [5.8, 3.1, 5.0, 1.7]], dtype=float)
y = list(classifier.predict(new_samples, as_iterable=True))
print('Predictions: {}'.format(str(y)))
'''