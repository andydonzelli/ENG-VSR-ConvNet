import TfCnnBase as base

import numpy as np
import tensorflow as tf

from scipy.io import savemat
import os
import sys

data_filename = '../TrainTest-Data/fullivs-data-(5,1).mat'

# Returns all subdirectories of b.
def all_subdirs_of(b='.'):
  result = []
  for d in os.listdir(b):
    bd = os.path.join(b, d)
    if os.path.isdir(bd): result.append(bd)
  return result


# Returns the name of the last subdirectory in b to be edited.
def GetLatestSubdir(b='.'):
    all_subdirs = all_subdirs_of(b)
    return max(all_subdirs, key=os.path.getmtime)


def main():
    # Get x and y from file.
    (x,y) = base.GetDataset(data_filename)
    x_test = base.Standardize(x_test)

    # If a model name was passed as an argument, use that. Otherwise, use the
    # last-edited subdirectory in models/.
    if len(sys.argv) == 1:
        model_dir = GetLatestSubdir('models/')
    else:
        model_dir = sys.argv[1]
    
    print('Using model in dir: ' + model_dir)
    
    # Load the model from the requested directory.
    classifier = tf.estimator.Estimator(model_fn=base.cnn_model_fn,
                                        model_dir=model_dir)

    # Use the model to make classifications (predictions) with input
    # from data_filename.
    predictions = list(classifier.predict(eval_input_fn))

    predicted_classes = [p['classes'] for p in predictions]
    predicted_probs = [p['probabilities'] for p in predictions]

    base.MeasurePerformance(predicted_classes, y_test)

    # Save the predictions (and associated confidence values) to file for
    # processing with Matlab.
    savemat('predictions-'+str(v_m), mdict={'pred_classes': predicted_classes,
                                            'pred_prob': predicted_probs})


if __name__ == '__main__':
    main()
