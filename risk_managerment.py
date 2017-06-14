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

import argparse
import sys
import tempfile

from six.moves import urllib

import pandas as pd
import tensorflow as tf

# Define variables to catas and cons. Catas are quantitative variables, for instance, gender. Cons are quantitative variables, for example, 10 and 20. 
COLUMNS = ["cata1", "cata2", "cata3", "cata4", "cata5", "cata6", "cata7", "cata8", "cata9", "cata10", "cata11",
           "con1", "con2", "con3", "con4", "con5", "con6", "con7", "con8", "con9", "y", "no" ]
LABEL_COLUMN = "label"
CATEGORICAL_COLUMNS = ["cata1", "cata2", "cata3", "cata4", "cata5", "cata6", "cata7", "cata8", "cata9", "cata10", "cata11"]
CONTINUOUS_COLUMNS = ["con1", "con2", "con3", "con4", "con5", "con6", "con7", "con8", "con9"]

def build_estimator(model_dir, model_type):
  """Build an estimator."""
  # Sparse base columns.
  cata1 = tf.contrib.layers.sparse_column_with_keys(column_name="cata1", keys=["female", "male"])
  cata2 = tf.contrib.layers.sparse_column_with_keys(column_name="cata2", keys=["A", "B", "C", "D", "E", "F"])
  cata3 = tf.contrib.layers.sparse_column_with_keys(column_name="cata3", keys=["APP", "ebusiness", "normal", "others"])
  cata4 = tf.contrib.layers.sparse_column_with_keys(column_name="cata4", keys=["Y", "N"])
  cata5 = tf.contrib.layers.sparse_column_with_keys(column_name="cata5", keys=["Y", "N"])
  cata6 = tf.contrib.layers.sparse_column_with_keys(column_name="cata6", keys=["Y", "N"])
  cata7 = tf.contrib.layers.sparse_column_with_keys(column_name="cata7", keys=["Y", "N"])
  cata8 = tf.contrib.layers.sparse_column_with_keys(column_name="cata8", keys=["Y", "N"])
  cata9 = tf.contrib.layers.sparse_column_with_keys(column_name="cata9", keys=["Y", "N"])
  cata10 = tf.contrib.layers.sparse_column_with_keys(column_name="cata10", keys=["Y", "N"])
  cata11 = tf.contrib.layers.sparse_column_with_keys(column_name="cata11", keys=["Y", "N"])

  # Continuous base columns.
  con1 = tf.contrib.layers.real_valued_column("con1")
  con2 = tf.contrib.layers.real_valued_column("con2")
  con3 = tf.contrib.layers.real_valued_column("con3")
  con4 = tf.contrib.layers.real_valued_column("con4")
  con5 = tf.contrib.layers.real_valued_column("con5")
  con6 = tf.contrib.layers.real_valued_column("con6")
  con7 = tf.contrib.layers.real_valued_column("con7")
  con8 = tf.contrib.layers.real_valued_column("con8")
  con9 = tf.contrib.layers.real_valued_column("con9")


  # Transformations.
  con1_buckets = tf.contrib.layers.bucketized_column(con1, boundaries=[99, 1000, 2000, 5000, 10000, 15000, 20000, 50000, 100000, 500000])
  con2_buckets = tf.contrib.layers.bucketized_column(con2, boundaries=[0, 1, 2, 3, 6, 9, 12, 16, 18, 24])
  con3_buckets = tf.contrib.layers.bucketized_column(con3, boundaries=[1, 5, 7, 10, 12, 16, 18, 20, 22, 24])
  con4_buckets = tf.contrib.layers.bucketized_column(con4, boundaries=[15, 20, 25, 30, 35, 40, 45, 50, 55, 60])
  con5_buckets = tf.contrib.layers.bucketized_column(con5, boundaries=[0, 1, 2, 3, 4, 5, 6, 8, 10, 20])
  con6_buckets = tf.contrib.layers.bucketized_column(con6, boundaries=[0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 10000, 20000])
  con7_buckets = tf.contrib.layers.bucketized_column(con7, boundaries=[0, 1000, 2000, 3000, 4000, 5000, 6000, 8000, 10000, 20000])
  con8_buckets = tf.contrib.layers.bucketized_column(con8, boundaries=[0, 1, 2, 3, 4, 5, 6, 8, 10, 20])
  con9_buckets = tf.contrib.layers.bucketized_column(con9, boundaries=[0, 1, 2, 3, 4, 5, 6, 8, 10, 20])
  
  # Wide columns and deep columns.
  wide_columns = [cata1, cata2, cata3, cata4, cata5, cata6, cata7, cata8, cata9, cata10, cata11,
                  con1_buckets, con2_buckets, con3_buckets, con4_buckets, con5_buckets, con6_buckets, con7_buckets, con8_buckets, con9_buckets]

  deep_columns = [
      tf.contrib.layers.embedding_column(cata1, dimension=8),
      tf.contrib.layers.embedding_column(cata2, dimension=8),
      tf.contrib.layers.embedding_column(cata3, dimension=8),
      tf.contrib.layers.embedding_column(cata4, dimension=8),
      tf.contrib.layers.embedding_column(cata5, dimension=8),
      tf.contrib.layers.embedding_column(cata6, dimension=8),
      tf.contrib.layers.embedding_column(cata7, dimension=8),
      tf.contrib.layers.embedding_column(cata8, dimension=8),
      tf.contrib.layers.embedding_column(cata9, dimension=8),
      tf.contrib.layers.embedding_column(cata10, dimension=8),
      tf.contrib.layers.embedding_column(cata11, dimension=8),

      con1,
      con2,
      con3,
      con4,
      con5,
      con6,
      con7,
      con8,
      con9]

  if model_type == "wide":
    m = tf.contrib.learn.LinearClassifier(model_dir=model_dir,
                                          feature_columns=wide_columns)
  elif model_type == "deep":
    m = tf.contrib.learn.DNNClassifier(model_dir=model_dir,
                                       feature_columns=deep_columns,
                                       hidden_units=[300, 250, 200, 150, 100, 50])
  else:
    m = tf.contrib.learn.DNNLinearCombinedClassifier(
        model_dir=model_dir,
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=[300, 250, 200, 150, 100, 50],
        fix_global_step_increment_bug=True)
  return m


def input_fn(df):
  """Input builder function."""
  # Creates a dictionary mapping from each continuous feature column name (k) to
  # the values of that column stored in a constant Tensor.
  continuous_cols = {k: tf.constant(df[k].values) for k in CONTINUOUS_COLUMNS}
  # Creates a dictionary mapping from each categorical feature column name (k)
  # to the values of that column stored in a tf.SparseTensor.
  categorical_cols = {
      k: tf.SparseTensor(
          indices=[[i, 0] for i in range(df[k].size)],
          values=df[k].values,
          dense_shape=[df[k].size, 1])
      for k in CATEGORICAL_COLUMNS}
  # Merges the two dictionaries into one.
  feature_cols = dict(continuous_cols)
  feature_cols.update(categorical_cols)
  # Converts the label column into a constant Tensor.
  label = tf.constant(df[LABEL_COLUMN].values)
  # Returns the feature columns and the label.
  return feature_cols, label


def train_and_eval(model_dir, model_type, train_steps, train_data, test_data):
  """Train and evaluate the model."""
  train_file_name = 'data/work_space_train.csv'
  test_file_name = 'data/work_space_test.csv'
#  train_file_name, test_file_name = maybe_download(train_data, test_data)
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

  # remove NaN elements
  df_train = df_train.dropna(how='any', axis=0)
  df_test = df_test.dropna(how='any', axis=0)

  df_train[LABEL_COLUMN] = (
      df_train["y"].apply(lambda x: "seal" in x)).astype(int)
  df_test[LABEL_COLUMN] = (
      df_test["y"].apply(lambda x: "seal" in x)).astype(int)

  model_dir = tempfile.mkdtemp() if not model_dir else model_dir
  print("model directory = %s" % model_dir)

  m = build_estimator(model_dir, model_type)
  m.fit(input_fn=lambda: input_fn(df_train), steps=train_steps)
  results = m.evaluate(input_fn=lambda: input_fn(df_test), steps=1)
  for key in sorted(results):
    print("%s: %s" % (key, results[key]))


FLAGS = None


def main(_):
  train_and_eval(FLAGS.model_dir, FLAGS.model_type, FLAGS.train_steps,
                 FLAGS.train_data, FLAGS.test_data)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument(
      "--model_dir",
      type=str,
      default="model/",
      help="Base directory for output models."
  )
  parser.add_argument(
      "--model_type",
      type=str,
      default="wide_n_deep",
      help="Valid model types: {'wide', 'deep', 'wide_n_deep'}."
  )
  parser.add_argument(
      "--train_steps",
      type=int,
      default=200,
      help="Number of training steps."
  )
  parser.add_argument(
      "--train_data",
      type=str,
      default="",
      help="Path to the training data."
  )
  parser.add_argument(
      "--test_data",
      type=str,
      default="",
      help="Path to the test data."
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)