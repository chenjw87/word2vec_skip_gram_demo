# a demo for training a word2vec skip-gram embedding
#====================================================

import tensorflow as tf
import reader
import math, random, pickle, collections
import numpy as np
from six.moves import xrange

# build dataset
def build_dataset(data_path):
	word_to_id = reader._build_vocab(data_path)
	data = reader._file_to_word_ids(data_path, word_to_id)
	id_to_word = dict(zip(word_to_id.values(), word_to_id.keys()))
	pickle.dump([word_to_id, id_to_word], open('data/vocab.pkl', "wb"))
	return data, word_to_id, id_to_word, len(word_to_id)

data, word_to_id, id_to_word, vocab_size = build_dataset('data/ptb.train.txt')
data_idx = 0

# generate a training batch for the skip-gram model
def generate_batch(batch_size, num_skips, skip_window):
	global data_idx
	assert num_skips <= 2 * skip_window
	assert batch_size % num_skips == 0
	batch = np.ndarray(shape=(batch_size), dtype=np.int32)
	labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32) # why 2-D
	span = 2 * skip_window + 1 # [skip_window target skip_window]
	if data_idx + span > len(data):
		data_idx = 0
	data_buffer = collections.deque(maxlen=span) # in order to implement a slide window on the context
	data_buffer.extend(data[data_idx:data_idx+span]) # extend an iterable list
	data_idx += span
	for i in range(batch_size // num_skips):
		# ecan target would be used num_skips to predict labels, which results in num_skips training pairs
		context_words = [w for w in range(span) if w != skip_window] # remove the id of the target
		words_to_use = random.sample(context_words, num_skips) # select num_skips for the target to predict
		for j, context_word in enumerate(words_to_use):
			# skip-gram is the revised version of CBOW
			batch[i * num_skips + j] = data_buffer[skip_window]
			labels[i * num_skips + j] = data_buffer[context_word]
		if data_idx == len(data):
			data_buffer = data[:span]
			data_idx = span
		else:
			data_buffer.append(data[data_idx]) # append a new element to the right. [a, b, c] -> [b, c, d]
			data_idx += 1
	# backtrack a little bit to avoid leaving words in the end of the batch underused as targets
	data_idx = (data_idx + len(data) - span) % len(data)
	return batch, labels

#batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)

# build and train a skip-gram model
batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1       # How many words to consider left and right.
num_skips = 2         # How many times to reuse an input to generate a label.
num_sampled = 64      # Number of negative examples to sample.

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent. These 3 variables are used only for
# displaying model accuracy, they don't affect calculation.
valid_size = 16     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)

graph = tf.Graph()
with graph.as_default():
	# inputs : batch of source context words, batch of target words
	train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
	train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1]) 
	valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

	# ops and variables pinned to the CPU
	with tf.device('/cpu:0'):
		# look up embeddings for inputs
		embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))
		embed = tf.nn.embedding_lookup(embeddings, train_inputs)

		# variables for logistic regression
		nce_weights = tf.Variable(tf.truncated_normal([vocab_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
		nce_biases = tf.Variable(tf.zeros([vocab_size]))

		# Compute the average NCE loss for the batch.
  		# tf.nce_loss automatically draws a new sample of the negative labels each
  		# time we evaluate the loss.
  		# http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
		loss = tf.reduce_mean(
					tf.nn.nce_loss(weights=nce_weights, 
									biases=nce_biases, 
									labels=train_labels, 
									num_sampled=num_sampled,
									num_classes=vocab_size,
									inputs=embed)
				)

  		# optimizer
		global_step = tf.Variable(0, trainable=False)
		start_learning_rate = 1.0
		lr = tf.train.exponential_decay(start_learning_rate, global_step, 10000, 0.96, staircase=True)
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss, global_step=global_step)

  		# initializer
		init = tf.global_variables_initializer()

# normalize each dense vector in the embeddings
norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
normalized_embeddings = embeddings / norm

# begin training
num_steps = 1000001 

with tf.Session(graph=graph) as session:
	# must initialize all variables before we use them
	init.run()
	print('initialize parameters...')

	average_loss = 0
	for step in xrange(num_steps):
		# load data into placeholder
		batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
		feed_dict = {train_inputs:batch_inputs, train_labels:batch_labels}

		_, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
		average_loss += loss_val

		if step % 2000 == 0:
			if step > 0:
				average_loss / 2000
			print('average loss at step ', step, ' : ', average_loss)
			average_loss = 0

	final_embeddings = normalized_embeddings.eval()

# save the embeddings matrix
pickle.dump(final_embeddings, open('data/embeddings.pkl', "wb"))