
# lets get the data by downloading it and creating a temp file
import tempfile
import urllib

train_file = tempfile.NamedTemporaryFile()
test_file = tempfile.NamedTemporaryFile()

urllib.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", train_file.name)
urllib.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test", test_file.name)

# lets get pandas to import the data from csv
import pandas as pd

COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num",
					 "marital_status", "occupation", "relationship", "race", "gender",
					 "income_bracket"]
					 
df_train = pd.read_csv(train_file, names=COLUMNS, skipinitialspace=True)
df_test = pd.read_csv(test_file, names=COLUMNS, skipinitialspace=True, skiprows=1)

# since binary classification we construct a label
# value is 1 if the income is over 50k
LABEL_COLUMN = "label"
df_train[LABEL_COLUMN] = (df_train["income_bracket"].apply(lambda x: ">50k" in x)).astype(int)
df_test[LABEL_COLUMN] = (df_test["income_bracket"].apply(lambda x: ">50k" in x)).astype(int)

# Groups where inputs fall in
CATEGORICAL_COLUMNS = ["workclass", "education", "marital_status", "occupation",
											"relationship", "race", "gender", "native_country"]
# inputs where a value lands in a range
CONTINUOUS_COLUMNS = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]

# ## Converting Data into Tensors

import tensorflow as tf

def input_fn(df):
	# Creates a dictionary mapping from each continuous feature column name (k) to
	# the values of that column stored in a constant Tensor.
	continuous = {k: tf.constant(df[k].values)
						for k in CONTINUOUS_COLUMNS}
					
	# Creates a dictionary mapping from each categorical feature column name (k)
	# to the values of that column stored in a tf.SparseTensor.
	categorical_cols = {k: tf.SparseTensor(
		indices=[[i,0] for i in range(df[k].size)],
		values=df[k].values,
		shape=[df[k].size, 1])
					for k in CATEGORICAL_COLUMNS}
					
	# merges the two dictionaries into one
	feature_cols = dict(continuous_cols.items() + categorical_cols.items())
	# Converts the label column into a constant Tensor
	label = tf.constant(df[LABEL_COLUMN].values)
	
	return feature_cols, label

def train_input_fn():
	return input_fn(df_train)
	
def eval_input_fn():
	return input_fn(df_test)

# ## Base Categorical Feature Columns
gender = tf.contrib.layers.sparse_column_with_keys(
	column_name="gender", keys=["Female", "Male"])
	
education = tf.contrib.layers.sparse_column_with_hash_bucket("education", hash_bucket_size=1000)

# categorical features
relationship = tf.contrib.layers.sparse_column_with_hash_bucket("relationship", hash_bucket_size=100)
workclass = tf.contrib.layers.sparse_column_with_hash_bucket("workclass", hash_bucket_size=100)
occupation = tf.contrib.layers.sparse_column_with_hash_bucket("occupation", hash_bucket_size=1000)
native_country = tf.contrib.layers.sparse_column_with_hash_bucket("native_country", hash_bucket_size=1000)
marital_status = tf.contrib.layers.sparse_column_with_hash_bucket("marital_status", hash_bucket_size=100)
race = tf.contrib.layers.sparse_column_with_hash_bucket("race", hash_bucket_size=100)

# Base Continuous Feature Columns
age = tf.contrib.layers.real_valued_column("age")
education_num = tf.contrib.layers.real_valued_column("education_num")
capital_gain = tf.contrib.layers.real_valued_column("capital_gain")
capital_loss = tf.contrib.layers.real_valued_column("capital_loss")
hours_per_week = tf.contrib.layers.real_valued_column("hours_per_week")

age_buckets = tf.contrib.layers.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

education_x_occupation = tf.contrib.layers.crossed_column([education, occupation], hash_bucket_size=int(1e4))

age_buckets_x_education_x_occupation = tf.contrib.layers.crossed_column( [age_buckets, education, occupation], hash_bucket_size=int(1e6) )

# ## Defining The Logistic Regression Model
model_dir = tempfile.mkdtemp()
m = tf.contrib.learn.LinearClassifier(feature_columns=[
	gender, native_country, education, occupation, workclass, marital_status, race,
	age_buckets, education_x_occupation, age_buckets_x_education_x_occupation],
	optimizer=tf.train.FtrlOptimizer(
	learning_rate=0.1,
	l1_regularization_strength=1.0,
	l2_regularization_strength=1.0),
	model_dir=model_dir)

# ## Training and Evaluating Our Model

m.fit(input_fn=train_input_fn, steps=200)

results = m.evaluate(input_fn=eval_input_fn, steps=1)

for key in sorted(results):
	if key == "capital_gain":
		continue
	print key, results[key]

















