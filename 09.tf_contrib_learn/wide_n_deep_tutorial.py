# -*- coding: utf-8 -*-

# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Example code for TensorFlow Wide & Deep Tutorial using TF.Learn API."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile
#import urllib
from six.moves import urllib

import pandas as pd
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("model_dir", "", "Base directory for output models.")
flags.DEFINE_string("model_type", "wide_n_deep",
                    "Valid model types: {'wide', 'deep', 'wide_n_deep'}.")
flags.DEFINE_integer("train_steps", 200, "Number of training steps.")
flags.DEFINE_string(
    "train_data",
    "",
    "Path to the training data.")
flags.DEFINE_string(
    "test_data",
    "",
    "Path to the test data.")

COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num",
           "marital_status", "occupation", "relationship", "race", "gender",
           "capital_gain", "capital_loss", "hours_per_week", "native_country",
           "income_bracket"]
LABEL_COLUMN = "label"
CATEGORICAL_COLUMNS = ["workclass", "education", "marital_status", "occupation",
                       "relationship", "race", "gender", "native_country"]
CONTINUOUS_COLUMNS = ["age", "education_num", "capital_gain", "capital_loss",
                      "hours_per_week"]


#################################################
# 学習,テスト用データダウンロード
#################################################
def maybe_download():
  """May be downloads training data and returns train and test file names."""
  if FLAGS.train_data:
    train_file_name = FLAGS.train_data
  else:
    train_file = tempfile.NamedTemporaryFile(delete=False)
    urllib.request.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", train_file.name)  # pylint: disable=line-too-long
    train_file_name = train_file.name
    train_file.close()
    print("Training data is downloaded to %s" % train_file_name)

  if FLAGS.test_data:
    test_file_name = FLAGS.test_data
  else:
    test_file = tempfile.NamedTemporaryFile(delete=False)
    urllib.request.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test", test_file.name)  # pylint: disable=line-too-long
    test_file_name = test_file.name
    test_file.close()
    print("Test data is downloaded to %s" % test_file_name)

  return train_file_name, test_file_name


#################################################
# Estimatorの初期化、設定
#################################################
def build_estimator(model_dir):
  """Build an estimator."""
  # Sparse base columns.

  # 連続値 : real_valued_column()使う
  # 名称とかでパターンが定まっているもの : sparse_column_with_keys()使う
  # 名称とかでパターンが定まっていないもの : sparse_column_with_hash_bucket()使う
  gender = tf.contrib.layers.sparse_column_with_keys(column_name="gender",
                                                     keys=["female", "male"])
  education = tf.contrib.layers.sparse_column_with_hash_bucket(
      "education", hash_bucket_size=1000)
  relationship = tf.contrib.layers.sparse_column_with_hash_bucket(
      "relationship", hash_bucket_size=100)
  workclass = tf.contrib.layers.sparse_column_with_hash_bucket(
      "workclass", hash_bucket_size=100)
  occupation = tf.contrib.layers.sparse_column_with_hash_bucket(
      "occupation", hash_bucket_size=1000)
  native_country = tf.contrib.layers.sparse_column_with_hash_bucket(
      "native_country", hash_bucket_size=1000)

  # Continuous base columns.
  age = tf.contrib.layers.real_valued_column("age")
  education_num = tf.contrib.layers.real_valued_column("education_num")
  capital_gain = tf.contrib.layers.real_valued_column("capital_gain")
  capital_loss = tf.contrib.layers.real_valued_column("capital_loss")
  hours_per_week = tf.contrib.layers.real_valued_column("hours_per_week")

  # Transformations.
  # 年齢を一定範囲でカテゴライズする。
  age_buckets = tf.contrib.layers.bucketized_column(age,
                                                    boundaries=[
                                                        18, 25, 30, 35, 40, 45,
                                                        50, 55, 60, 65
                                                    ])

  # Wide columns and deep columns.
  # gender, native_country, education, occupation, workclass, relationship, age_buckets,
  # [education, occupation], [age_buckets, education, occupation], [native_country, occupation]
  wide_columns = [gender, native_country, education, occupation, workclass,
                  relationship, age_buckets,
                  tf.contrib.layers.crossed_column([education, occupation],
                                                   hash_bucket_size=int(1e4)),
                  tf.contrib.layers.crossed_column(
                      [age_buckets, education, occupation],
                      hash_bucket_size=int(1e6)),
                  tf.contrib.layers.crossed_column([native_country, occupation],
                                                   hash_bucket_size=int(1e4))]
  deep_columns = [
      tf.contrib.layers.embedding_column(workclass, dimension=8),
      tf.contrib.layers.embedding_column(education, dimension=8),
      tf.contrib.layers.embedding_column(gender, dimension=8),
      tf.contrib.layers.embedding_column(relationship, dimension=8),
      tf.contrib.layers.embedding_column(native_country,
                                         dimension=8),
      tf.contrib.layers.embedding_column(occupation, dimension=8),
      age,
      education_num,
      capital_gain,
      capital_loss,
      hours_per_week,
  ]

  # 実行時引数でmodel_typeに"wide"が指定されたら"LinearClassifier",
  # "deep"が指定された"DNNClassifier"を使用する。
  if FLAGS.model_type == "wide":
    m = tf.contrib.learn.LinearClassifier(model_dir=model_dir,
                                          feature_columns=wide_columns)
  elif FLAGS.model_type == "deep":
    m = tf.contrib.learn.DNNClassifier(model_dir=model_dir,
                                       feature_columns=deep_columns,
                                       hidden_units=[100, 50])
  else:
    m = tf.contrib.learn.DNNLinearCombinedClassifier(
        model_dir=model_dir,
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=[100, 50])
  return m


#################################################
# 入力データ設定
#################################################
def input_fn(df):
  """Input builder function."""
  # Creates a dictionary mapping from each continuous feature column name (k) to
  # the values of that column stored in a constant Tensor.
  # No列作成(1〜データ数の自然数の列を作成)
  continuous_cols = {k: tf.constant(df[k].values) for k in CONTINUOUS_COLUMNS}

  # Creates a dictionary mapping from each categorical feature column name (k)
  # to the values of that column stored in a tf.SparseTensor.
  #   indices: A 2-D int64 tensor of shape [N, ndims].
  #   values: A 1-D tensor of any type and shape [N].
  #   shape: A 1-D int64 tensor of shape [ndims].
  # kはディクショナリのキー。CATEGORICAL_COLUMNSから取得。
  # df自体がディクショナリ。
  categorical_cols = {k: tf.SparseTensor(
      indices=[[i, 0] for i in range(df[k].size)],
      values=df[k].values,
      shape=[df[k].size, 1])
                      for k in CATEGORICAL_COLUMNS}

  # Merges the two dictionaries into one.
  # ディクショナリ合体。
  feature_cols = dict(continuous_cols)
  feature_cols.update(categorical_cols)

  # Converts the label column into a constant Tensor.
  # ラベル列をTensorに変換。
  # ラベル列：収入が50,000以上か否か
  label = tf.constant(df[LABEL_COLUMN].values)

  # Returns the feature columns and the label.
  return feature_cols, label


#################################################
# 学習、テスト
#################################################
def train_and_eval():
  """Train and evaluate the model."""

  # 学習、テスト用データ取得
  train_file_name, test_file_name = maybe_download()

  # ダウンロードした学習、テスト用データ(CSV)を読み込む。
  #   names : List of column names to use. If file contains no header row, then you should explicitly pass header=None.
  #   skipinitialspace : Skip spaces after delimiter.
  #   engine : Parser engine to use. The C engine is faster while the python engine is currently more feature-complete.
  df_train = pd.read_csv(
      tf.gfile.Open(train_file_name),
      names=COLUMNS,
      skipinitialspace=True,
      engine="python")
  df_test = pd.read_csv(
      tf.gfile.Open(test_file_name),
      names=COLUMNS,
      skipinitialspace=True,
      skiprows=1,
      engine="python")

  # 収入が50,000以上なら1, それ以外なら0のデータをlabel columnに設定
  df_train[LABEL_COLUMN] = (
      df_train["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)
  df_test[LABEL_COLUMN] = (
      df_test["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)

  # モデルのディレクトリ設定
  # FLAS.model_dirがNoneならmkdtemp()でファイル名を生成する。
  model_dir = tempfile.mkdtemp() if not FLAGS.model_dir else FLAGS.model_dir
  print("model directory = %s" % model_dir)

  # Estimatorの初期化、設定
  m = build_estimator(model_dir)

  # 学習
  #   m : LinearClassifier or DNNClassifier or DNNLinearCombinedClassifier
  m.fit(input_fn=lambda: input_fn(df_train), steps=FLAGS.train_steps)

  # テスト
  #   resultsの中身は自動？
  results = m.evaluate(input_fn=lambda: input_fn(df_test), steps=1)
  for key in sorted(results):
    print("%s: %s" % (key, results[key]))


#################################################
# main
#################################################
def main(_):
  train_and_eval()

#################################################
# エントリーポイント
#################################################
if __name__ == "__main__":
  tf.app.run()
