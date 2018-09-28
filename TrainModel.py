from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import TfCnnBase as base
import sys

batch_size = 32
epochs = 3
vm = 10
data_filename = '../TrainTest-Data/data10-(1,0)-HR(3,17)-shf.mat'

tf.logging.set_verbosity(tf.logging.INFO)


def PrepareXY(x,y):
    # Coherentely shuffle the data set and labels.
    x, y, _ = base.PermuteXY(x,y)

    # Add samples to class-balance the data sets.
    x, y = base.OversampleDataset(x,y)

    # Coherently shuffle the data set and labels.
    x, y, _ = base.PermuteXY(x,y)

    print('Number of samples: {}'.format(len(y)))
    print('Proportion of 1s: {}\n'.format(np.sum(y)/len(y)))

    x = base.Standardize(x)

    # Separate x and y into training and testing subsets.
    th_idx = int(0.9*len(y))
    x_train = x[0:th_idx]
    x_test = x[th_idx:]
    y_train = y[0:th_idx]
    y_test = y[th_idx:]

    return (x_train, x_test, y_train, y_test)


def main(unused_argv):
    dir_name = 'cnnfinal-{}ep-{}vm'.format(epochs, vm)

    print('using dir_name: {}'.format(dir_name))
    print('using filename: {}'.format(data_filename))

    # Get x and y (data and labels) from Mat file, + processing.
    (x,y) = base.GetDataset(data_filename)
    (x_train, _, y_train, _) = PrepareXY(x,y)

    # Create the Estimator
    classifier = tf.estimator.Estimator(model_fn=base.cnn_model_fn,
                                        model_dir='models/'+dir_name,
                                        config=tf.estimator.RunConfig(
                                                        save_checkpoints_steps=1000,
                                                        save_checkpoints_secs=None,
                                                        save_summary_steps=1000,
                                                        keep_checkpoint_max=1))

    # Set up logging for predictions
    # Log the values in the 'Softmax' tensor with label 'probabilities'
    tensors_to_log = {'probabilities': 'softmax_tensor'}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log,
                                              every_n_iter=50000)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(x={'x': x_train},
                                                        y=y_train,
                                                        batch_size=batch_size,
                                                        num_epochs=epochs,
                                                        shuffle=False)

    classifier.train(input_fn=train_input_fn, hooks=[logging_hook])

    # Save the model (for later testing or use).
    classifier.export_savedmodel('models/'+dir_name,
                                 base.serving_input_receiver_fn)


if __name__ == '__main__':
    tf.app.run()
