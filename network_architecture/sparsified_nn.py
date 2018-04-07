import tensorflow as tf
import numpy as np
from network_architecture.network import conv
from network_architecture.network import lc
from network_architecture.network import mpool
from network_architecture.network import fc


class SparsifiedNN:
    _channels = 3
    _nPools = 4
    _dropping_matrices = dict()
    _sparse_layer = ""

    def __init__(self, channels, height, width, num_of_classes=512, dropping_matrices=dict(), sparse_layer=""):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self._channels = channels
            self._dropping_matrices = dropping_matrices
            self._sparse_layer = sparse_layer

            stddev = np.sqrt(2. / (3 * 3 * self._channels))
            self.weights_1a = tf.Variable(tf.truncated_normal([3, 3, self._channels, 64], stddev=stddev))
            self.biases_1a = tf.Variable(tf.zeros([64]))

            stddev = np.sqrt(2. / (3 * 3 * 64))
            self.weights_1b = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=stddev))
            self.biases_1b = tf.Variable(tf.zeros([64]))

            stddev = np.sqrt(2. / (3 * 3 * 64))
            self.weights_2a = tf.Variable(tf.truncated_normal([3, 3, 64, 96], stddev=stddev))
            self.biases_2a = tf.Variable(tf.zeros([96]))

            stddev = np.sqrt(2. / (3 * 3 * 96))
            self.weights_2b = tf.Variable(tf.truncated_normal([3, 3, 96, 96], stddev=stddev))
            self.biases_2b = tf.Variable(tf.zeros([96]))

            stddev = np.sqrt(2. / (3 * 3 * 96))
            self.weights_3a = tf.Variable(tf.truncated_normal([3, 3, 96, 192], stddev=stddev))
            self.biases_3a = tf.Variable(tf.zeros([192]))

            stddev = np.sqrt(2. / (3 * 3 * 192))
            self.weights_3b = tf.Variable(tf.truncated_normal([3, 3, 192, 192], stddev=stddev))
            self.biases_3b = tf.Variable(tf.zeros([192]))

            stddev = np.sqrt(2. / (3 * 3 * 192))
            self.weights_4a = tf.Variable(tf.truncated_normal([3, 3, 192, 256], stddev=stddev))
            self.biases_4a = tf.Variable(tf.zeros([256]))

            stddev = np.sqrt(2. / (3 * 3 * 256))
            self.weights_4b = tf.Variable(tf.truncated_normal([3, 3, 256, 256], stddev=stddev))
            self.biases_4b = tf.Variable(tf.zeros([256]))

            new_height = height / pow(2, self._nPools)
            new_width = width / pow(2, self._nPools)
            output_row = int((new_height - 3) / 1 + 1)
            output_col = int((new_width - 3) / 1 + 1)

            weight_row = output_row * output_col
            weight_column = 3 * 3 * 256
            stddev = np.sqrt(2. / (weight_row * weight_column))

            self.weights_lc_5a = tf.Variable(tf.truncated_normal([weight_row, weight_column, 256], stddev=stddev))
            self.biases_lc_5a = tf.Variable(tf.zeros([1, output_row, output_col, 256]))

            new_height = output_row
            new_width = output_col
            output_row = int((new_height - 3) / 1 + 1)
            output_col = int((new_width - 3) / 1 + 1)

            weight_row = output_row * output_col
            weight_column = 3 * 3 * 256
            stddev = np.sqrt(2. / (weight_row * weight_column))

            self.weights_lc_5b = tf.Variable(tf.truncated_normal([weight_row, weight_column, 256], stddev=stddev))
            self.biases_lc_5b = tf.Variable(tf.zeros([1, output_row, output_col, 256]))

            input_size = output_row * output_col * 256
            stddev = np.sqrt(2. / input_size)

            self.weights_fc = tf.Variable(tf.truncated_normal([input_size, num_of_classes], stddev=stddev))
            self.biases_fc = tf.Variable(tf.zeros([num_of_classes]))


    def get_graph(self):
        return self.graph


    def forward_pass(self, data, reuse):
        dropping_matrix = self._get_dropping_matrix('conv1a')
        if dropping_matrix is not None:
            self.weights_1a = tf.multiply(self.weights_1a, dropping_matrix)
        self.conv1a = conv(data, self.weights_1a, self.biases_1a, 1, 1, 'SAME', 'conv1a', reuse)

        dropping_matrix = self._get_dropping_matrix('conv1b')
        if dropping_matrix is not None:
            self.weights_1b = tf.matmul(self.weights_1b, dropping_matrix)
        self.conv1b = conv(self.conv1a, self.weights_1b, self.biases_1b, 1, 1, 'SAME', 'conv1b', reuse)

        self.pool1 = mpool(self.conv1b, 2, 2, 2, 2, 'SAME', 'mpool1', reuse)

        dropping_matrix = self._get_dropping_matrix('conv2a')
        if dropping_matrix is not None:
            self.weights_2a = tf.multiply(self.weights_2a, dropping_matrix)
        self.conv2a = conv(self.pool1, self.weights_2a, self.biases_2a, 1, 1, 'SAME', 'conv2a', reuse)

        dropping_matrix = self._get_dropping_matrix('conv2b')
        if dropping_matrix is not None:
            self.weights_2b = tf.multiply(self.weights_2b, dropping_matrix)
        self.conv2b = conv(self.conv2a, self.weights_2b, self.biases_2b, 1, 1, 'SAME', 'conv2b', reuse)

        self.pool2 = mpool(self.conv2b, 2, 2, 2, 2, 'SAME', 'mpool2', reuse)

        dropping_matrix = self._get_dropping_matrix('conv3a')
        if dropping_matrix is not None:
            self.weights_3a = tf.multiply(self.weights_3a, dropping_matrix)
        self.conv3a = conv(self.pool2, self.weights_3a, self.biases_3a, 1, 1, 'SAME', 'conv3a', reuse)

        dropping_matrix = self._get_dropping_matrix('conv3b')
        if dropping_matrix is not None:
            self.weights_3b = tf.multiply(self.weights_3b, dropping_matrix)
        self.conv3b = conv(self.conv3a, self.weights_3b, self.biases_3b, 1, 1, 'SAME', 'conv3b', reuse)

        self.pool3 = mpool(self.conv3b, 2, 2, 2, 2, 'SAME', 'mpool3', reuse)

        dropping_matrix = self._get_dropping_matrix('conv4a')
        if dropping_matrix is not None:
            self.weights_4a = tf.multiply(self.weights_4a, dropping_matrix)
        self.conv4a = conv(self.pool3, self.weights_4a, self.biases_4a, 1, 1, 'SAME', 'conv4a', reuse)

        dropping_matrix = self._get_dropping_matrix('conv4b')
        if dropping_matrix is not None:
            self.weights_4b = tf.multiply(self.weights_4b, dropping_matrix)
        self.conv4b = conv(self.conv4a, self.weights_4b, self.biases_4b, 1, 1, 'SAME', 'conv4b', reuse)

        self.pool4 = mpool(self.conv4b, 2, 2, 2, 2, 'SAME', 'mpool4', reuse)

        dropping_matrix = self._get_dropping_matrix('lc5a')
        if dropping_matrix is not None:
            self.weights_lc_5a = tf.multiply(self.weights_lc_5a, dropping_matrix)
        self.lc5a = lc(self.pool4, self.weights_lc_5a.get_shape().as_list()[2], self.weights_lc_5a, self.biases_lc_5a, 3, 3, 1, 1, 'lc5a', reuse)

        # for prelogits1
        shape = self.lc5a.get_shape().as_list()
        lc5a_reshape = tf.reshape(self.lc5a, [-1, shape[1] * shape[2] * shape[3]])

        dropping_matrix = self._get_dropping_matrix('lc5b')
        if dropping_matrix is not None:
            self.weights_lc_5b = tf.multiply(self.weights_lc_5b, dropping_matrix)
        self.lc5b = lc(self.lc5a, self.weights_lc_5b.get_shape().as_list()[2], self.weights_lc_5b, self.biases_lc_5b, 3, 3, 1, 1, 'lc5b', reuse)

        # for prelogits2
        shape = self.lc5b.get_shape().as_list()
        lc5b_reshape = tf.reshape(self.lc5b, [-1, shape[1] * shape[2] * shape[3]])

        dropping_matrix = self._get_dropping_matrix('fc')
        if dropping_matrix is not None:
            self.weights_fc = tf.multiply(self.weights_fc, dropping_matrix)
        # for logits
        self.fc = fc(self.lc5b, self.weights_fc, self.biases_fc, 'fc', reuse)

        return lc5a_reshape, lc5b_reshape, self.fc


    def _get_dropping_matrix(self, name):
        for value in self._dropping_matrices.values():
            if value[0] == name:
                return value[1]

        return None

    def get_layer(self, layer_name, is_previous=False):
        if layer_name == 'conv1a':
            return self.conv1a
        if layer_name == 'conv1b':
            if is_previous:
                return self.pool1
            return self.conv1b
        if layer_name == 'conv2a':
            return self.conv2a
        if layer_name == 'conv2b':
            if is_previous:
                return self.pool2
            return self.conv2b
        if layer_name == 'conv3a':
            return self.conv3a
        if layer_name == 'conv3b':
            if is_previous:
                return self.pool3
            return self.conv3b
        if layer_name == 'conv4a':
            return self.conv4a
        if layer_name == 'conv4b':
            if is_previous:
                return self.pool4
            return self.conv4b
        if layer_name == 'lc5a':
            return self.lc5a
        if layer_name == 'lc5b':
            return self.lc5b
        if layer_name == 'fc':
            return self.fc


    def get_weights(self, layer_name):
        if layer_name == 'conv1a':
            return self.weights_1a
        if layer_name == 'conv1b':
            return self.weights_1b
        if layer_name == 'conv2a':
            return self.weights_2a
        if layer_name == 'conv2b':
            return self.weights_2b
        if layer_name == 'conv3a':
            return self.weights_3a
        if layer_name == 'conv3b':
            return self.weights_3b
        if layer_name == 'conv4a':
            return self.weights_4a
        if layer_name == 'conv4b':
            return self.weights_4b
        if layer_name == 'lc5a':
            return self.weights_lc_5a
        if layer_name == 'lc5b':
            return self.weights_lc_5b
        if layer_name == 'fc':
            return self.weights_fc
