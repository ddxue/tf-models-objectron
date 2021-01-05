# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Efficient Objectron input pipeline using tf.data.Dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections
import functools
import os
import numpy as np

from absl import logging
import six
import tensorflow.compat.v1 as tf

from efficientnet import preprocessing
from objectron.schema import features


def build_image_serving_input_fn(image_size,
                                 batch_size=None,
                                 resize_method=None):
  """Builds a serving input fn for raw images."""

  def _image_serving_input_fn():
    """Serving input fn for raw images."""

    def _preprocess_image(image_bytes):
      """Preprocess a single raw image."""
      image = preprocessing.preprocess_image(
          image_bytes=image_bytes,
          is_training=False,
          image_size=image_size,
          resize_method=resize_method)
      return image

    image_bytes_list = tf.placeholder(
        shape=[batch_size],
        dtype=tf.string,
    )
    images = tf.map_fn(
        _preprocess_image, image_bytes_list, back_prop=False, dtype=tf.float32)
    return tf.estimator.export.ServingInputReceiver(
        images, {'image_bytes': image_bytes_list})
  return _image_serving_input_fn


class ObjectronTFExampleInput(six.with_metaclass(abc.ABCMeta, object)):
  """Base class for Objectron input_fn generator."""

  def __init__(self,
               is_training,
               use_bfloat16,
               num_cores=8,
               image_size=224,
               num_parallel_calls=24,
               transpose_input=False,
               num_label_classes=18,
               include_background_label=False,
               augment_name=None,
               mixup_alpha=0.0,
               randaug_num_layers=None,
               randaug_magnitude=None,
               resize_method=None):
    """Constructor.

    Args:
      is_training: `bool` for whether the input is for training
      use_bfloat16: If True, use bfloat16 precision; else use float32.
      num_cores: `int` for the number of TPU cores
      image_size: `int` for image size (both width and height).
      num_parallel_calls: concurrency level to use when reading data from disk.
      transpose_input: 'bool' for whether to use the double transpose trick
      num_label_classes: number of label classes. Default to 18 for Objectron.
      include_background_label: If true, label #0 is reserved for background.
      augment_name: `string` that is the name of the augmentation method to
        apply to the image. `autoaugment` if AutoAugment is to be used or
        `randaugment` if RandAugment is to be used. If the value is `None` no no
        augmentation method will be applied applied. See autoaugment.py for more
        details.
      mixup_alpha: float to control the strength of Mixup regularization, set to
        0.0 to disable.
      randaug_num_layers: 'int', if RandAug is used, what should the number of
        layers be. See autoaugment.py for detailed description.
      randaug_magnitude: 'int', if RandAug is used, what should the magnitude
        be. See autoaugment.py for detailed description.
      resize_method: If None, use bicubic in default.
    """
    self.image_preprocessing_fn = preprocessing.preprocess_image
    self.is_training = is_training
    self.use_bfloat16 = use_bfloat16
    self.num_cores = num_cores
    self.transpose_input = transpose_input
    self.image_size = image_size
    self.num_parallel_calls = num_parallel_calls
    self.include_background_label = include_background_label
    self.num_label_classes = num_label_classes
    if include_background_label:
      self.num_label_classes += 1
    self.augment_name = augment_name
    self.mixup_alpha = mixup_alpha
    self.randaug_num_layers = randaug_num_layers
    self.randaug_magnitude = randaug_magnitude
    self.resize_method = resize_method
    self.num_channels = 3

  def set_shapes(self, batch_size, images, labels):
    """Statically set the batch_size dimension."""
    if self.transpose_input:
      images.set_shape(images.get_shape().merge_with(
          tf.TensorShape([None, None, None, batch_size])))
      labels.set_shape(labels.get_shape().merge_with(
          tf.TensorShape([batch_size, None])))
      # Convert to R1 tensors for fast transfer to device.
      images = tf.reshape(images, [-1])
    else:
      images.set_shape(images.get_shape().merge_with(
          tf.TensorShape([batch_size, None, None, None])))
      labels.set_shape(labels.get_shape().merge_with(
          tf.TensorShape([batch_size, None])))

    return images, labels

  def mixup(self, batch_size, alpha, images, labels):
    """Applies Mixup regularization to a batch of images and labels.

    [1] Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, David Lopez-Paz
      Mixup: Beyond Empirical Risk Minimization.
      ICLR'18, https://arxiv.org/abs/1710.09412

    Arguments:
      batch_size: The input batch size for images and labels.
      alpha: Float that controls the strength of Mixup regularization.
      images: A batch of images of shape [batch_size, ...]
      labels: A batch of labels of shape [batch_size, num_classes]

    Returns:
      A tuple of (images, labels) with the same dimensions as the input with
      Mixup regularization applied.
    """
    mix_weight = tf.distributions.Beta(alpha, alpha).sample([batch_size, 1])
    mix_weight = tf.maximum(mix_weight, 1. - mix_weight)
    images_mix_weight = tf.cast(
        tf.reshape(mix_weight, [batch_size, 1, 1, 1]), images.dtype)
    # Mixup on a single batch is implemented by taking a weighted sum with the
    # same batch in reverse.
    images_mix = (
        images * images_mix_weight + images[::-1] * (1. - images_mix_weight))
    labels_mix = labels * mix_weight + labels[::-1] * (1. - mix_weight)
    return images_mix, labels_mix

  def dataset_parser(self, value):
    """Parses an image and its label from a serialized TFExample.

    Args:
      value: serialized string containing an Objectron TFExample.

    Returns:
      Returns a tuple of (image, label) from the TFExample.
    """
    # The 3D bounding box has 9 vertices, 0: is the center, and the 8 vertices of the 3D box.
    NUM_KEYPOINTS = 9
    NUM_CHANNELS = 3

    parsed = tf.parse_single_example(value, features = features.FEATURE_MAP)
    image_bytes = tf.reshape(parsed[features.FEATURE_NAMES['IMAGE_ENCODED']], shape=[])
    original_shape = tf.convert_to_tensor([
      tf.reshape(parsed[features.FEATURE_NAMES['IMAGE_HEIGHT']], shape=[]),
      tf.reshape(parsed[features.FEATURE_NAMES['IMAGE_WIDTH']], shape=[]),
      NUM_CHANNELS,
    ], dtype=tf.int32)
    
    image = self.image_preprocessing_fn(
        image_bytes=image_bytes,
        original_shape=original_shape,
        is_training=self.is_training,
        image_size=self.image_size,
        use_bfloat16=self.use_bfloat16,
        augment_name=self.augment_name,
        randaug_num_layers=self.randaug_num_layers,
        randaug_magnitude=self.randaug_magnitude,
        resize_method=self.resize_method)

    # import pdb; pdb.set_trace()
    # number_objects_batch is a tensor of shape (batch-size,) which tells the 
    # number of objects in each batch slice.
    num_objects = tf.reduce_sum(parsed[features.FEATURE_NAMES['INSTANCE_NUM']])
    keypoints = tf.reshape(parsed[features.FEATURE_NAMES['POINT_2D']].values, [num_objects, NUM_KEYPOINTS, 3])

    # The object annotation is a list of 3x1 keypoints for all the annotated
    # objects. The objects can have a varying number of keypoints. First we split
    # the list according to the number of keypoints for each object. This
    # also leaves an empty array at the end of the list.
    # object_keypoints = tf.split(keypoints, num_objects)

    # The keypoints are [x, y, d] where `x` and `y` are normalized (`uv`-system)\
    # and `d` is the metric distance from the center of the camera. Convert them
    # keypoint's `xy` value to pixel.
    first_keypoints = keypoints[0, :, :2]

    # Flatten the labels into a (18,) tensor representing the 9x2 normalized keypoints.
    label = tf.reshape(first_keypoints, shape=[-1])
    print(label)
    return image, label

  @abc.abstractmethod
  def make_source_dataset(self, index, num_hosts):
    """Makes dataset of serialized TFExamples.

    The returned dataset will contain `tf.string` tensors, but these strings are
    serialized `TFExample` records that will be parsed by `dataset_parser`.

    If self.is_training, the dataset should be infinite.

    Args:
      index: current host index.
      num_hosts: total number of hosts.

    Returns:
      A `tf.data.Dataset` object.
    """
    return

  def input_fn(self, params):
    """Input function which provides a single batch for train or eval.

    Args:
      params: `dict` of parameters passed from the `TPUEstimator`.
          `params['batch_size']` is always provided and should be used as the
          effective batch size.

    Returns:
      A `tf.data.Dataset` object.
    """
    # Retrieves the batch size for the current shard. The # of shards is
    # computed according to the input pipeline deployment. See
    # tf.estimator.tpu.RunConfig for details.
    batch_size = params['batch_size']

    if 'context' in params:
      current_host = params['context'].current_input_fn_deployment()[1]
      num_hosts = params['context'].num_hosts
    else:
      current_host = 0
      num_hosts = 1

    dataset = self.make_source_dataset(current_host, num_hosts)

    # Use the fused map-and-batch operation.
    #
    # For XLA, we must used fixed shapes. Because we repeat the source training
    # dataset indefinitely, we can use `drop_remainder=True` to get fixed-size
    # batches without dropping any training examples.
    #
    # When evaluating, `drop_remainder=True` prevents accidentally evaluating
    # the same image twice by dropping the final batch if it is less than a full
    # batch size. As long as this validation is done with consistent batch size,
    # exactly the same images will be used.
    dataset = dataset.map(self.dataset_parser, self.num_parallel_calls).batch(batch_size, True)

    # Apply Mixup
    if self.is_training and self.mixup_alpha > 0.0:
      dataset = dataset.map(
          functools.partial(self.mixup, batch_size, self.mixup_alpha),
          num_parallel_calls=self.num_parallel_calls)

    # Transpose for performance on TPU
    if self.transpose_input:
      dataset = dataset.map(
          lambda images, labels: (tf.transpose(images, [1, 2, 3, 0]), labels),
          num_parallel_calls=self.num_parallel_calls)

    # Assign static batch size dimension
    dataset = dataset.map(functools.partial(self.set_shapes, batch_size), self.num_parallel_calls)

    # Prefetch overlaps in-feed with training
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    options = tf.data.Options()
    options.experimental_deterministic = False
    options.experimental_threading.max_intra_op_parallelism = 1
    options.experimental_threading.private_threadpool_size = 48
    dataset = dataset.with_options(options)

    return dataset


class ObjectronInput(ObjectronTFExampleInput):
  """Generates Objectron input_fn from a series of TFRecord files.

  The training data is assumed to be in TFRecord format with keys as specified
  in the dataset_parser below, sharded across 1012 files, named sequentially:

      shoe_train-00000-of-01012
      shoe_train-00001-of-01012
      ...
      shoe_train-01011-of-01012

  The evaluation data is in the same format but sharded in 128 files.

  The format of the data required is created by the script at:
      https://github.com/tensorflow/tpu/blob/master/tools/datasets/imagenet_to_gcs.py
  """

  def __init__(self,
               is_training,
               use_bfloat16,
               transpose_input,
               data_dir,
               image_size=224,
               num_parallel_calls=24,
               cache=False,
               num_label_classes=18,
               include_background_label=False,
               augment_name=None,
               mixup_alpha=0.0,
               randaug_num_layers=None,
               randaug_magnitude=None,
               resize_method=None,
               holdout_shards=None):
    """Create an input from TFRecord files.

    Args:
      is_training: `bool` for whether the input is for training
      use_bfloat16: If True, use bfloat16 precision; else use float32.
      transpose_input: 'bool' for whether to use the double transpose trick
      data_dir: `str` for the directory of the training and validation data;
          if 'null' (the literal string 'null') or implicitly False
          then construct a null pipeline, consisting of empty images
          and blank labels.
      image_size: `int` for image size (both width and height).
      num_parallel_calls: concurrency level to use when reading data from disk.
      cache: if true, fill the dataset by repeating from its cache.
      num_label_classes: number of label classes. Default to 18 for Objectron.
      include_background_label: if true, label #0 is reserved for background.
      augment_name: `string` that is the name of the augmentation method
          to apply to the image. `autoaugment` if AutoAugment is to be used or
          `randaugment` if RandAugment is to be used. If the value is `None` no
          no augmentation method will be applied applied. See autoaugment.py
          for more details.
      mixup_alpha: float to control the strength of Mixup regularization, set
          to 0.0 to disable.
      randaug_num_layers: 'int', if RandAug is used, what should the number of
        layers be. See autoaugment.py for detailed description.
      randaug_magnitude: 'int', if RandAug is used, what should the magnitude
        be. See autoaugment.py for detailed description.
      resize_method: If None, use bicubic in default.
      holdout_shards: number of holdout training shards for validation.
    """
    super(ObjectronInput, self).__init__(
        is_training=is_training,
        image_size=image_size,
        use_bfloat16=use_bfloat16,
        transpose_input=transpose_input,
        num_label_classes=num_label_classes,
        include_background_label=include_background_label,
        augment_name=augment_name,
        mixup_alpha=mixup_alpha,
        randaug_num_layers=randaug_num_layers,
        randaug_magnitude=randaug_magnitude)
    self.data_dir = data_dir
    if self.data_dir == 'null' or not self.data_dir:
      self.data_dir = None
    self.num_parallel_calls = num_parallel_calls
    self.cache = cache
    self.holdout_shards = holdout_shards
    self.num_train_shards = 1012

  def _get_null_input(self, data):
    """Returns a null image (all black pixels).

    Args:
      data: element of a dataset, ignored in this method, since it produces
          the same null image regardless of the element.

    Returns:
      a tensor representing a null image.
    """
    del data  # Unused since output is constant regardless of input
    return tf.zeros([self.image_size, self.image_size, 3], tf.bfloat16
                    if self.use_bfloat16 else tf.float32)

  def dataset_parser(self, value):
    """See base class."""
    if not self.data_dir:
      return value, tf.constant(0., tf.float32, (18,))
    return super(ObjectronInput, self).dataset_parser(value)

  def make_source_dataset(self, index, num_hosts):
    """See base class."""
    if not self.data_dir:
      logging.info('Undefined data_dir implies null input')
      return tf.data.Dataset.range(1).repeat().map(self._get_null_input)

    if self.holdout_shards:
      if self.is_training:
        filenames = [
            os.path.join(self.data_dir, 'shoe_train-%05d-of-%05d' % (i, self.num_train_shards))
            for i in range(self.holdout_shards, self.num_train_shards)
        ]
      else:
        filenames = [
            os.path.join(self.data_dir, 'shoe_train-%05d-of-%05d' % (i, self.num_train_shards))
            for i in range(0, self.holdout_shards)
        ]
      for f in filenames[:10]:
        logging.info('datafiles: %s', f)
      dataset = tf.data.Dataset.from_tensor_slices(filenames)
    else:
      file_pattern = os.path.join(
          self.data_dir, 'shoe_train-*' if self.is_training else 'shoe_test-*')
      logging.info('datafiles: %s', file_pattern)
      dataset = tf.data.Dataset.list_files(file_pattern, shuffle=False)

    # For multi-host training, we want each hosts to always process the same
    # subset of files.  Each host only sees a subset of the entire dataset,
    # allowing us to cache larger datasets in memory.
    dataset = dataset.shard(num_hosts, index)

    if self.is_training and not self.cache:
      dataset = dataset.repeat()

    def fetch_dataset(filename):
      buffer_size = 1 * 1024 * 1024  # 8 MiB per file
      dataset = tf.data.TFRecordDataset(filename, buffer_size=buffer_size)
      return dataset

    # Read the data from disk in parallel
    dataset = dataset.interleave(
        fetch_dataset, cycle_length=self.num_parallel_calls,
        num_parallel_calls=self.num_parallel_calls, deterministic=False)

    if self.cache:
      dataset = dataset.cache().shuffle(128 * 16).repeat()
    else:
      dataset = dataset.shuffle(128)
    return dataset


# Defines a selection of data from a Cloud Bigtable.
BigtableSelection = collections.namedtuple('BigtableSelection', [
    'project', 'instance', 'table', 'prefix', 'column_family',
    'column_qualifier'
])


class ObjectronBigtableInput(ObjectronTFExampleInput):
  """Generates Objectron input_fn from a Bigtable for training or evaluation.
  """

  def __init__(self,
               is_training,
               use_bfloat16,
               transpose_input,
               selection,
               augment_name=None,
               num_label_classes=18,
               include_background_label=False,
               mixup_alpha=0.0,
               randaug_num_layers=None,
               randaug_magnitude=None,
               resize_method=None):
    """Constructs an Objectron input from a BigtableSelection.

    Args:
      is_training: `bool` for whether the input is for training
      use_bfloat16: If True, use bfloat16 precision; else use float32.
      transpose_input: 'bool' for whether to use the double transpose trick
      selection: a BigtableSelection specifying a part of a Bigtable.
      augment_name: `string` that is the name of the augmentation method
          to apply to the image. `autoaugment` if AutoAugment is to be used or
          `randaugment` if RandAugment is to be used. If the value is `None` no
          no augmentation method will be applied applied. See autoaugment.py
          for more details.
      num_label_classes: number of label classes. Default to 18 for Objectron.
      include_background_label: if true, label #0 is reserved for background.
      mixup_alpha: float to control the strength of Mixup regularization, set
          to 0.0 to disable.
      randaug_num_layers: 'int', if RandAug is used, what should the number of
        layers be. See autoaugment.py for detailed description.
      randaug_magnitude: 'int', if RandAug is used, what should the magnitude
        be. See autoaugment.py for detailed description.s
      resize_method: if None, use bicubic.
    """
    super(ObjectronBigtableInput, self).__init__(
        is_training=is_training,
        use_bfloat16=use_bfloat16,
        transpose_input=transpose_input,
        num_label_classes=num_label_classes,
        include_background_label=include_background_label,
        augment_name=augment_name,
        mixup_alpha=mixup_alpha,
        randaug_num_layers=randaug_num_layers,
        randaug_magnitude=randaug_magnitude,
        resize_method=resize_method)
    self.selection = selection

  def make_source_dataset(self, index, num_hosts):
    """See base class."""
    try:
      from tensorflow.contrib.cloud import BigtableClient  # pylint: disable=g-import-not-at-top
    except ImportError as e:
      logging.exception('Bigtable is not supported in TensorFlow 2.x.')
      raise e

    data = self.selection
    client = BigtableClient(data.project, data.instance)
    table = client.table(data.table)
    ds = table.parallel_scan_prefix(data.prefix,
                                    columns=[(data.column_family,
                                              data.column_qualifier)])
    # The Bigtable datasets will have the shape (row_key, data)
    ds_data = ds.map(lambda index, data: data)

    if self.is_training:
      ds_data = ds_data.repeat()

    return ds_data
