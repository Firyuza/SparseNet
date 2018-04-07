import tensorflow as tf
from tensorflow.python.ops import control_flow_ops


def batch_norm(x, phase_train, reuse=None):
    """
    Batch normalization on convolutional maps.
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Variable, true indicates training phase
        scope:       string, variable scope
        affn:      whether to affn-transform outputs
    Return:
        normed:      batch-normalized maps
    Ref: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow/33950177
    """
    name = 'batch_norm'
    with tf.variable_scope(name, reuse=reuse):
        phase_train = tf.convert_to_tensor(phase_train, dtype=tf.bool)
        n_out = int(x.get_shape()[3])
        beta = tf.Variable(tf.constant(0.0, shape=[n_out], dtype=x.dtype),
                           name=name+'/beta', trainable=True, dtype=x.dtype)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out], dtype=x.dtype),
                            name=name+'/gamma', trainable=True, dtype=x.dtype)

        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.9)
        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)
        mean, var = control_flow_ops.cond(phase_train,
                                          mean_var_with_update,
                                          lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed

def conv(inpOp, kernel, biases, dH, dW, padType, name, reuse):
    with tf.variable_scope(name, reuse=reuse):
        conv_bn = tf.nn.conv2d(inpOp, kernel, [1, dH, dW, 1], padding=padType)
        bias = tf.nn.bias_add(conv_bn, biases)
        output = tf.nn.relu(bias)

    return output

def lc(inpOp, nOut, kernel, biases, kH, kW, dH, dW, name, reuse):
    with tf.variable_scope(name, reuse=reuse):
        shape = inpOp.get_shape().as_list()
        nIn = shape[3]
        output_row = int((shape[1] - kW) / dW + 1)
        output_col = int((shape[2] - kH) / dH + 1)
        xs = []
        """for i in range(output_row):
            for j in range(output_col):
                xs.append(tf.reshape(
                    inpOp[:, i * dW:i * dW + kW, j * dH:j * dH + kH, :],
                    (1, -1, kH * kW * nIn)))"""

        [[xs.append(tf.reshape(inpOp[:, i * dW:i * dW + kW, j * dH:j * dH + kH, :],(1, -1, kH * kW * nIn))) for j in range(output_col)] for i in range(output_row)]

        x_aggregate = tf.concat(xs, 0)
        lc = tf.matmul(x_aggregate, kernel)
        lc = tf.reshape(lc, (output_row, output_col, -1, nOut))
        lc = tf.transpose(lc, (2, 0, 1, 3))
        bias = tf.add(lc, biases)
        output = tf.nn.relu(bias)

    return output

def mpool(input, kH, kW, dH, dW, padding, name, reuse):
    with tf.variable_scope(name, reuse=reuse):
        maxpool = tf.nn.max_pool(input,
                       ksize=[1, kH, kW, 1],
                       strides=[1, dH, dW, 1],
                       padding=padding)

    return maxpool

def fc(inpOp, weights, biases, name, reuse):
    with tf.variable_scope(name, reuse=reuse):
        shape = inpOp.get_shape().as_list()
        dim = 1
        for d in shape[1:]:
            dim *= d
        x = tf.reshape(inpOp, [-1, dim])
        fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

    return fc
