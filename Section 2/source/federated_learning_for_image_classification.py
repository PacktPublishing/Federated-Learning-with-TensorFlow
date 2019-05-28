# -*- coding: utf-8 -*-
# Based on the original code example:
# https://www.tensorflow.org/federated/tutorials/federated_learning_for_image_classification
# Simplified, added an example of random client sampling.

import collections
import numpy as np
np.random.seed(0)

import tensorflow as tf
from tensorflow.python.keras.optimizer_v2 import gradient_descent

from tensorflow_federated import python as tff
from random import choices

NUM_EPOCHS = 5
BATCH_SIZE = 20
SHUFFLE_BUFFER = 500
NUM_CLIENTS = 3

tf.compat.v1.enable_v2_behavior()

# Loading simulation data
emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()

def preprocess(dataset):
  def element_fn(element):
    return collections.OrderedDict([
        ('x', tf.reshape(element['pixels'], [-1])),
        ('y', tf.reshape(element['label'], [1])),
    ])

  return dataset.repeat(NUM_EPOCHS).map(element_fn).shuffle(
      SHUFFLE_BUFFER).batch(BATCH_SIZE)


def make_federated_data(client_data, client_ids):
  return [preprocess(client_data.create_tf_dataset_for_client(x))
          for x in client_ids]

sample_clients = emnist_train.client_ids[0: NUM_CLIENTS]
federated_train_data = make_federated_data(emnist_train, sample_clients)

sample_clients_test = emnist_test.client_ids[0: NUM_CLIENTS]
federated_test_data = make_federated_data(emnist_test, sample_clients_test)

# This is only needed to create the "federated" ver of the model
sample_batch = iter(federated_train_data[0]).next()
sample_batch = collections.OrderedDict([
    ('x', sample_batch['x'].numpy()),
    ('y', sample_batch['y'].numpy()),
])

# Training

# Create a new model
def create_compiled_keras_model():
  model = tf.keras.models.Sequential([
      tf.keras.layers.Dense(
          10, activation=tf.nn.softmax, kernel_initializer='zeros', input_shape=(784,))])

  def loss_fn(y_true, y_pred):
    return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(
        y_true, y_pred))

  model.compile(
      loss=loss_fn,
      optimizer=gradient_descent.SGD(learning_rate=0.02),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
  return model

# Turn model into one that can be used with TFF
def model_fn():
  keras_model = create_compiled_keras_model()
  return tff.learning.from_compiled_keras_model(keras_model, sample_batch)

# Initialize training
iterative_process = tff.learning.build_federated_averaging_process(model_fn)
state = iterative_process.initialize()

trained_clients=[]
def get_train_data(keep_it_stupid_simple=False):
    if keep_it_stupid_simple:
        if not trained_clients:
            trained_clients.append(sample_clients)
        return federated_train_data
    sc = choices(emnist_train.client_ids, k=NUM_CLIENTS)
    for c in sc:
        while True:
            if c in trained_clients:
                sc.remove(c)
                newc=choices(emnist_train.client_ids, k=1)[0]
                if newc not in trained_clients:
                    sc.append(newc)
                    break
            else:
                break
    trained_clients.append(sc)
    new_federated_train_data = make_federated_data(emnist_train, sc)
    return new_federated_train_data

# Training process
for round_num in range(1, NUM_EPOCHS+1):
  federated_train_data=get_train_data(True)
  state, metrics = iterative_process.next(state, federated_train_data)
  print('round {:2d}, metrics={}'.format(round_num, metrics))

print('Trained {:2d} clients'.format(len(trained_clients)*NUM_CLIENTS))
print(trained_clients)

# Evaluation
evaluation = tff.learning.build_federated_evaluation(model_fn)

train_metrics = evaluation(state.model, federated_train_data)
print('Train metrics', str(train_metrics))

test_metrics = evaluation(state.model, federated_test_data)
print('Test metrics', str(test_metrics))
