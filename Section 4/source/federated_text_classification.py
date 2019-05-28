# -*- coding: utf-8 -*-
# Federated learning for text classification based on the tensorflow tutorial
# https://www.tensorflow.org/tutorials/keras/basic_text_classification
VOCAB_SIZE = 10000
SEQ_LENGTH = 256
BATCH_SIZE = 512
EPOCHS=3
SHUFFLE_BUFFER=5000
CLIENTS=3

import tensorflow as tf
from tensorflow import keras
import collections

import numpy as np

from tensorflow.python.keras.optimizer_v2 import gradient_descent
from tensorflow_federated import python as tff

nest = tf.contrib.framework.nest

np.random.seed(0)

tf.compat.v1.enable_v2_behavior()

# Data preparation
imdb = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))

print(train_data[0])
print(train_labels[0])

# A dictionary mapping words to an integer index
word_index = imdb.get_word_index()

# The first indices are reserved
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=SEQ_LENGTH)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=SEQ_LENGTH)

def preprocess(dataset):
  def element_fn(x, y):
    return collections.OrderedDict([
        ('x', x),
        ('y', tf.cast(tf.reshape(y, [1]), tf.float32))
    ])

  return dataset.map(element_fn).shuffle(
      SHUFFLE_BUFFER).batch(BATCH_SIZE)

def generate_clients_datasets(n, source_x, source_y):
    clients_dataset=[]
    for i in range(n+1):
        dataset=tf.data.Dataset.from_tensor_slices(([source_x[i]], [source_y[i]]))
        dataset=preprocess(dataset)
        clients_dataset.append(dataset)
    return clients_dataset

train_dataset=generate_clients_datasets(CLIENTS, train_data, train_labels)
test_dataset=generate_clients_datasets(CLIENTS, test_data, test_labels)

# Grab a single batch of data so that TFF knows what data looks like.
sample_batch = tf.contrib.framework.nest.map_structure(
    lambda x: x.numpy(), iter(train_dataset[0]).next())


# Createing and preparing the model
def create_compiled_keras_model():
    model = keras.Sequential()
    model.add(keras.layers.Embedding(VOCAB_SIZE, 16))
    model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.Dense(16, activation=tf.nn.relu))
    model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
    def loss_fn(y_true, y_pred):
        return tf.reduce_mean(
        tf.keras.metrics.binary_crossentropy(
            y_true, y_pred, from_logits=True))
    model.compile(optimizer=gradient_descent.SGD(learning_rate=0.02),
              loss=loss_fn, metrics=['acc'])
    return model

def model_fn():
  keras_model = create_compiled_keras_model()
  return tff.learning.from_compiled_keras_model(keras_model, sample_batch)

# Training and evaluating the model
iterative_process = tff.learning.build_federated_averaging_process(model_fn)
state = iterative_process.initialize()

for n in range(EPOCHS):
    state, metrics = iterative_process.next(state, train_dataset)
    print('round  {}, training metrics={}'.format(n+1, metrics))

evaluation = tff.learning.build_federated_evaluation(model_fn)
eval_metrics = evaluation(state.model, train_dataset)
print('Training evaluation metrics={}'.format(eval_metrics))

test_metrics = evaluation(state.model, test_dataset)
print('Test evaluation metrics={}'.format(test_metrics))
