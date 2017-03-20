import tensorflow as tf

# ## Define Base Feature Columns
# Categorical base column where values fall into a group
gender = tf.contrib.layers.sparse_column_with_keys(column_name="gender", keys=["Female", "Male"])
race = tf.contrib.layers.sparse_column_with_keys(column_name="race", keys=[
	"Amer-Indian-Eskimos", "Asian-Pac-Islander", "Black", "Other", "White"])
education = tf.contrib.layers.sparse_column_with_hash_bucket("education", hash_bucket_size=1000)
relationship = tf.contrib.layers.sparse_column_with_hash_bucket("relationship", hash_bucket_size=100)
workclass = tf.contrib.layers.sparse_column_with_hash_bucket("workclass", hash_bucket_size=100)
occupation = tf.contrib.layers.sparse_column_with_hash_bucket("occupation", hash_bucket_size=1000)
native_country = tf.contrib.layers.sparse_column_with_hash_bucket("native_country", hash_bucket_size=1000)

# Continuous base columns where the values fall into a number ranve
age = tf.contrib.layers.real_valued_column("age")
age_buckets = tf.contrib.layers.bucketized_column(age, boundaries=[18, 25, 30, 40, 45, 50, 55, 60, 65])
education_num = tf.contrib.layers.real_valued_column("education_num")
capital_gain  = tf.contrib.layers.real_valued_column("capital_gain")
capital_loss  = tf.contrib.layers.real_valued_column("capital_loss")
hours_per_week  = tf.contrib.layers.real_valued_column("hours_per_week")


# ## The Wide Model: Linear Model with Crossed Feature Columns
# wide column are columns which are associated with other columns to give more meaning
wide_columns = [
	gender, native_country, education, occupation, workclass, relationship, age_buckets,
		tf.contrib.layers.crossed_column([education, occupation], hash_bucket_size=int(1e4)),
		tf.contrib.layers.crossed_column([native_country, occupation], hash_bucket_size=int(1e4)),
		tf.contrib.layers.crossed_column([age_buckets, education, occupation], hash_bucket_size=int(1e6))]

# ## The Deep Model: Neural Network with Embeddings 
# will create a deep model and give the high demension features and reduce them to lower demensions
deep_columns = [
	tf.contrib.layers.embedding_column(workclass, dimension=8),
	tf.contrib.layers.embedding_column(education, dimension=8),
	tf.contrib.layers.embedding_column(gender, dimension=8),
	tf.contrib.layers.embedding_column(relationship, dimension=8),
	tf.contrib.layers.embedding_column(native_country, dimension=8),
	tf.contrib.layers.embedding_column(occupation, dimension=8),
	age, education_num, capital_gain, capital_loss, hours_per_week]
	
# ## Combining Wide and Deep Models into One
import tempfile
model_dir = tempfile.mkdtemp()
# merge a wide and deep model hybrid
m = tf.contrib.learn.DNNLinearCombinedClassifier(
	model_dir=model_dir,
	linear_feature_columns=wide_columns,
	dnn_feature_columns=deep_columns,
	dnn_hidden_units=[100, 50])
	
# ## Training and Evaluating The Model
import pandas as pd
import urllib

# define column for data set
COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num",
	"marital_status", "occupation", "relationship", "race", "gender",
	"capital_gain", "capital_loss", "hours_per_week", "native_country", "income_bracket"]
LABEL_COLUMN = 'label'
CATEGORICAL_COLUMNS = ["workclass", "education", "marital_status", "occupation",
	"relationship", "race", "gender", "native_country"]
CONTINUOUS_COLUMNS = ["age", "education_num", "capital_gain", "capital_loss",
	"hours_per_week"]

# Download our data files
train_file = tempfile.NamedTemporaryFile()
test_file = tempfile.NamedTemporaryFile()
urllib.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", train_file.name)
urllib.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test", test_file.name)

# load the data from csv files
df_train = pd.read_csv(train_file, names=COLUMNS, skipinitialspace=True)
df_test = pd.read_csv(test_file, names=COLUMNS, skipinitialspace=True, skiprows=1)
df_train[LABEL_COLUMN] = (df_train['income_bracket'].apply(lambda x: '>50K' in x)).astype(int)
df_test[LABEL_COLUMN] = (df_test['income_bracket'].apply(lambda x: '>50K' in x)).astype(int)

# input the data
def input_fn(df):
	continuous_cols = {k: tf.constant(df[k].values)
								for k in CONTINUOUS_COLUMNS}
	
	categorical_cols = {k: tf.SparseTensor(
		indices=[[i, 0] for i in range(df[k].size)],
		values=df[k].values,
		shape=[df[k].size, 1])
								for k in CATEGORICAL_COLUMNS}
	feature_cols = dict(continuous_cols.items() + categorical_cols.items())
	label = tf.constant(df[LABEL_COLUMN].values)
								
	return feature_cols, label
# train with data
def train_input_fn():
	return input_fn(df_train)
# test with data
def eval_input_fn():
	return input_fn(df_test)

# train the model
m.fit(input_fn=train_input_fn, steps=200)
# evaluate the model 
results = m.evaluate(input_fn=eval_input_fn, steps=1)
for key in sorted(results):
	print "%s: %s" % (key, results[key]) 

