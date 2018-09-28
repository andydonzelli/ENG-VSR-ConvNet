from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import h5py

img_rows = 5
img_cols = 375

# Given a Matlab-created string stored in a H5 format (deep array), returns a
# python string equivalent.
def GetH5String(arr):
    s = ''
    for c in arr:
        s += chr(c[0])
    return s


# Reads values in the Params structure of the mat file and uses these to
# determine the "image" rows and columns.
def GetImgDims(data_filename):
    # Extract required parameters from Param in mat file.
    with h5py.File(data_filename, 'r') as f:
        num_electrodes = f['Params']['NumElectrodes'][0][0]
        ch_comb_type = GetH5String(f['Params']['ChannelCombType'])
        sig_width = f['Params']['SignalWidth'][0][0]
        sample_period = f['Params']['SamplingPeriod'][0][0]

    if ch_comb_type == 'unipolar':
        numCh = num_electrodes
    elif ch_comb_type == 'bipolar':
        numCh = num_electrodes - 1
    elif ch_comb_type == 'tripolar':
        numCh = num_electrodes - 2

    img_rows = int(numCh)
    img_cols = int(round((sig_width/sample_period)+1e-15))
    
    return (img_rows, img_cols)


# Reads values in the Params structure of the mat file and uses these to
# determine the Matched Velocity field.
def GetMatchedVelocity(data_filename):
    # Extract required parameters from Param in mat file.
    with h5py.File(data_filename, 'r') as f:
        v_m = f['Params']['MatchedVelocity'][0][0]

    return v_m


# Given an NxM array, returns a new NxM array where each length-M vector has been
# standardized to zero mean and unit variance.
def Standardize(x):
    return ((x.transpose() - np.mean(x, axis=1)) / np.std(x, axis=1)).transpose()


# Coherentely shuffles the elements in arrays x and y.
def PermuteXY(x,y):
    p = np.random.permutation(len(y))
    return (x[p], y[p], p)


# Removes the first N class-0 samples required to achieve class-balance.
def UndersampleDataset(x,y):
    num_samples = len(y)
    del_idx = []
    count = len(y) - 2*np.sum(y)
    for i in range(len(y)):
        if y[i] == 0:
            del_idx.append(i)
            count -= 1
            if count == 0:
                break

    x = np.delete(x, del_idx, axis=0)
    y = np.delete(y, del_idx, axis=0)
    print('Discarded {} samples to obtain class balance.'.format(num_samples-len(y)))
    return (x,y)


# Replicates the first N class-1 samples required to achieve class-balance.
def OversampleDataset(x,y):
    num_samples = len(y)
    rep_idx = []
    count = len(y) - 2*np.sum(y)
    while(count != 0):
        for i in range(len(y)):
            if y[i] == 1:
                rep_idx.append(i)
                count -= 1
                if count == 0:
                    break

    x = np.concatenate((x, x[rep_idx]), axis=0)
    y = np.concatenate((y, y[rep_idx]), axis=0)
    print('Added {} samples to obtain class balance.'.format(len(y)-num_samples))
    return (x,y)


# Extracts the data samples (x) and associated labels (y) from file.
def GetDataset(filename='../TrainTest-Data/data.mat'):
    with h5py.File(filename, 'r') as f:
        x = np.array(f['dataSignal'])
        y = np.array(f['labelSignal'])

    x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
    y = y.reshape(len(y),)
    y = y.astype(np.int32)

    return (x, y)


# The model definition of the CNN structure.
def cnn_model_fn(features, labels, mode):
    input_layer = tf.reshape(features['x'], [-1, img_rows, img_cols, 1])

    conv1 = tf.layers.conv2d(inputs=input_layer,
                             filters=32,
                             kernel_size=[1, 10],
                             padding='valid',
                             activation=tf.nn.relu,
                             name='conv1')


    pool2_flat = tf.reshape(conv1, [-1, 32 * img_rows * (img_cols-9)])
    dense = tf.layers.dense(inputs=pool2_flat, units=256,
                            activation=tf.nn.relu, name='dense1')

    logits = tf.layers.dense(inputs=dense, units=2,
                             activation=tf.nn.relu, name='logits')
    
    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        'classes': tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }

    export_outputs = {'y': tf.estimator.export.PredictOutput(predictions)}

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=predictions,
                                          export_outputs=export_outputs)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        accuracy = tf.metrics.accuracy(labels, predictions['classes'])
        tf.summary.scalar('accuracy', accuracy[1])

        # Calculate True Negative rate.
        true_negatives = tf.metrics.true_negatives(labels,
                                                   predictions['classes'])[1]
        false_positives = tf.metrics.false_positives(labels,
                                                     predictions['classes'])[1]
        tf.summary.scalar('true_negative_rate',
                                true_negatives/(true_negatives+false_positives))

        # Calculate True Positive rate.
        true_positives = tf.metrics.true_positives(labels,
                                                   predictions['classes'])[1]
        false_negatives = tf.metrics.false_negatives(labels,
                                                     predictions['classes'])[1]
        tf.summary.scalar('true_positive_rate',
                                true_positives/(true_positives+false_negatives))

        optimizer = tf.train.AdadeltaOptimizer()
        train_op = optimizer.minimize(loss=loss,
                                      global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          train_op=train_op,
                                          export_outputs=export_outputs)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(labels, predictions['classes'])
    }

    return tf.estimator.EstimatorSpec(mode=mode,
                                      loss=loss,
                                      eval_metric_ops=eval_metric_ops,
                                      export_outputs={'y':
                                    tf.estimator.export.PredictOutput(predictions)})


def serving_input_receiver_fn():
    feature_spec = {'x': tf.placeholder(tf.float32, [None,img_rows*img_cols,])}
    return tf.estimator.export.build_raw_serving_input_receiver_fn(feature_spec)()


# Prints the True Positive and Negative rates based on a set of predictions and labels
def MeasurePerformance(predicted_classes, y_test):
    num_label_1 = np.sum(y_test)
    num_label_0 = len(y_test) - num_label_1

    true_positives = 0
    true_negatives = 0

    for i in range(len(predicted_classes)):
        if predicted_classes[i] == y_test[i] == 0:
            true_negatives += 1
        elif predicted_classes[i] == y_test[i] == 1:
            true_positives += 1

    print('\nTPr: {}%'.format(100.0 * true_positives/num_label_1))
    print('TNr: {}%\n'.format(100.0 * true_negatives/num_label_0))