'''
Let's learn about the linear models tf learn has for us
'''

# ##Encoding sparse columns

# categorical features in linear models are typically translated into sparse vectors
eye_color = tf.contrib.layers.sparse_column_with_keys(
	column_name="eye_color", keys=["blue", "brown", "green"])

# for categorical features for which you don't know all possible values
education = tf.contrib.layers.sparse_column_with_hash_bucket(
	"education", hash_bucket_size=1000)
	
# ##Feature Crosses

# if two features have a relationship enforce the relationship 
# by creating another feature witch is called feature cross
sport = tf.contrib.layers.sparse_column_with_hash_bucket(
	"sport", hash_bucket_size=1000)
city = tf.contrib.layers.sparse_column_with_hash_bucket(
	"city", hash_bucket_size=1000)
sport_x_city = tf.contrib.layers.crossed_column(
	[sport, city], hash_bucket_size=int(1e4))
	
# add another feature with real_valued_column
age = tf.contrib.layers.real_valued_column("age")

# ## Bucketization

# divides the range pf possible values into subranges called buckets
age_buckets = tf.contrib.layers.bucketized_column(
	age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 50, 65])
	

# ## Linear estimators
e = tf.contrib.learn.LinearClassifier( feature_columns = [
	native_country, education, occupation, workclass, marital_status,
	race,, age_buckets, education_x_occupation, age_buckets_x_race_x_occupation],
	model_dir=YOUR_MODEL_DIRECTORY)
e.fit(input_fn=input_fn_train, steps=200)

# Evaluate for one step (one pass through the test data)
results = e.evaluate(input_fn=input_fn_test, steps=1)

# lets see the stats
for key in sorted(results):
	print "%s: %s" % (key, results[key]) 


# ## Wide and deep learning

e = tf.contrib.learn.DNNLinearCombinedClassifier(
	model_dir=YOUR_MODEL_DIRECTORY,
	linear_feature_columns=wide_column,
	dnn_feature_columns=deep_columns,
	dnn_hidden_units=[100, 50]
)