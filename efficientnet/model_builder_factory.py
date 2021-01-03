# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Utilities for model builder or input size."""

from efficientnet import efficientnet_builder
from efficientnet.lite import efficientnet_lite_builder


def get_model_builder(model_name):
  """Get the model_builder module for a given model name."""
  if model_name.startswith('efficientnet-lite'):
    return efficientnet_lite_builder
  elif model_name.startswith('efficientnet-'):
    return efficientnet_builder
  else:
    raise ValueError(
        'Model must be either efficientnet-b* or efficientnet-lite*')


def get_model_input_size(model_name):
  """Get model input size for a given model name."""
  if model_name.startswith('efficientnet-lite'):
    _, _, image_size, _ = (
        efficientnet_lite_builder.efficientnet_lite_params(model_name))
  elif model_name.startswith('efficientnet'):
    _, _, image_size, _ = efficientnet_builder.efficientnet_params(model_name)
  else:
    raise ValueError(
        'Model must be either efficientnet-b* or efficientnet-lite*')
  return image_size
