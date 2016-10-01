# -*- coding: utf-8 -*-

#################################################
# tf.contrib.learn Quickstart
#################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

# Data sets
IRIS_TRAINING = "iris_training.csv"
IRIS_TEST = "iris_test.csv"

#################################################
# エントリーポイント
# [Args]:
#################################################
if __name__ == '__main__':

    # Load datasets.
    training_set = tf.contrib.learn.datasets.base.load_csv(filename=IRIS_TRAINING,
                                                           target_dtype=np.int)
    test_set = tf.contrib.learn.datasets.base.load_csv(filename=IRIS_TEST,
                                                       target_dtype=np.int)

    # Specify that all features have real-value data0
    # 萼片の幅、萼片の高さ、花弁の幅、花弁の高さの4次元データ
    # (sepal width, sepal height, petal width, and petal height)
    feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]

    # Build 3 layer DNN with 10, 20, 10 units respectively.
    classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                                hidden_units=[10, 20, 10],
                                                n_classes=3,
                                                model_dir="/tmp/iris_model")

    # Fit model.
    print('--------------------------------------------------')
    print('Start fitting.')
    classifier.fit(x=training_set.data,
                   y=training_set.target,
                   steps=2000)

    # Evaluate accuracy.
    accuracy_score = classifier.evaluate(x=test_set.data,
                                         y=test_set.target)["accuracy"]
    print('--------------------------------------------------')
    print('Accuracy: {0:f}'.format(accuracy_score))

    # Classify two new flower samples.
    # 新しい2サンプルを判定。正解は[1 2]らしい。
    new_samples = np.array(
        [[6.4, 3.2, 4.5, 1.5], [5.8, 3.1, 5.0, 1.7]], dtype=float)
    y = classifier.predict(new_samples)
    print('--------------------------------------------------')
    print('Predictions: {}'.format(str(y)))