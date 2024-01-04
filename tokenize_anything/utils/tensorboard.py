# ------------------------------------------------------------------------
# Copyright (c) 2023-present, BAAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
"""Tensorboard application."""

import time

import numpy as np

try:
    import tensorflow as tf
except ImportError:
    tf = None


class TensorBoard(object):
    """TensorBoard application."""

    def __init__(self, log_dir=None):
        """Create a summary writer logging to log_dir."""
        if tf is None:
            raise ImportError("Failed to import ``tensorflow`` package.")
        tf.config.set_visible_devices([], "GPU")
        if log_dir is None:
            log_dir = "./logs/" + time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
        self.writer = tf.summary.create_file_writer(log_dir)

    @staticmethod
    def is_available():
        """Return if tensor board is available."""
        return tf is not None

    def close(self):
        """Close board and apply all cached summaries."""
        self.writer.close()

    def histogram_summary(self, tag, values, step, buckets=10):
        """Write a histogram of values."""
        with self.writer.as_default():
            tf.summary.histogram(tag, values, step, buckets=buckets)

    def image_summary(self, tag, images, step, order="BGR"):
        """Write a list of images."""
        if isinstance(images, (tuple, list)):
            images = np.stack(images)
        if len(images.shape) != 4:
            raise ValueError("Images can not be packed to (N, H, W, C).")
        if order == "BGR":
            images = images[:, :, :, ::-1]
        with self.writer.as_default():
            tf.summary.image(tag, images, step, max_outputs=images.shape[0])

    def scalar_summary(self, tag, value, step):
        """Write a scalar."""
        with self.writer.as_default():
            tf.summary.scalar(tag, value, step)
