import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops


class Batch:
    _paths = []
    _image_height = 0
    _image_width = 0
    _height_offset = 0
    _width_offset = 0
    _batch_size = 0
    _num_labels = 0
    _labels = []

    def __init__(self, paths, image_height, image_width, height_offset, width_offset, batch_size, num_labels, labels):
        self._paths = paths
        self._image_height = image_height
        self._image_width = image_width
        self._height_offset = height_offset
        self._width_offset = width_offset
        self._batch_size = batch_size
        self._num_labels = num_labels
        self._labels = labels

    def read_data(self, nrof_preprocess_threads, seed, phase_train, label_smoothing, r_mean=0, g_mean=0, b_mean=0):
        self._paths = ops.convert_to_tensor(self._paths, dtype=tf.string)

        if phase_train:
            labels = ops.convert_to_tensor(self._labels, dtype=tf.int32)
            labels = tf.one_hot(labels, self._num_labels, 1.0 - label_smoothing,
                                label_smoothing / (self._num_labels - 1))
            input_queue = tf.train.slice_input_producer([self._paths, labels], shuffle=True, seed=seed)
            images_and_labels = []
            for _ in range(nrof_preprocess_threads):
                # Following a previous convention, each pixel (in [0, 255]) in RGB images is normalized by subtracting 127.5 then dividing by 128.
                # horisontal flip for center loss function The faces are cropped to 112 x 96 RGB images.
                label = input_queue[1]
                image = self._read_and_augment_image(input_queue[0], augment=True, seed=seed, r_mean=r_mean, g_mean=g_mean, b_mean=b_mean)
                images_and_labels.append([image, label])
            image_batch, label_batch = tf.train.batch_join(images_and_labels, self._batch_size)
            return image_batch, label_batch
        else:
            input_queue = tf.train.slice_input_producer([self._paths], shuffle=False, seed=seed)
            images = []
            image = self._read_and_augment_image(input_queue[0], augment=False, seed=seed, r_mean=r_mean, g_mean=g_mean, b_mean=b_mean)
            images.append([image])
            image_batch = tf.train.batch_join(images, self._batch_size)

            return image_batch

    def _read_and_augment_image(self, filename, augment, seed, r_mean, g_mean, b_mean):
        file_contents = tf.read_file(filename)
        image = tf.image.decode_png(file_contents, channels=3)
        image = tf.image.resize_images(image, [self._image_height + self._height_offset,
                                               self._image_width + self._width_offset])
        if augment:
            image = tf.random_crop(image, [self._image_height, self._image_width, 3], seed=seed)
            image = tf.image.random_flip_left_right(image, seed)
        else:
            image = tf.image.crop_to_bounding_box(image, self._height_offset // 2, self._width_offset // 2,
                                                  self._image_height,

                                                  self._image_width)
        image = (image - 127.0) / 128.

        return image
