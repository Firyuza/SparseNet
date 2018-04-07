import tensorflow as tf

class Loss:

    @staticmethod
    def get_center_loss(features, labels, alpha, num_labels):

        nrof_features = features.get_shape()[1]

        centers = tf.get_variable('centers', [num_labels, nrof_features], dtype=tf.float32,
            initializer=tf.constant_initializer(0.0), trainable=False)

        labels = tf.argmax(labels, 1)
        labels = tf.reshape(labels, [-1])

        centers_batch = tf.gather(centers, labels)
        diff = (1 - alpha) * (centers_batch - features)

        centers = tf.scatter_sub(centers, labels, diff)
        loss = tf.nn.l2_loss(features - centers_batch)
        return loss, centers

    @staticmethod
    def get_verif_loss(embeddings, labels, alpha):
        nrof_rows = embeddings.get_shape()[0]
        embeddings_matrix = tf.stack([embeddings] * nrof_rows)
        embeddings_matrix_t = tf.transpose(embeddings_matrix, [1, 0, 2])

        nrof_rows = labels.get_shape()[0]
        labels_matrix = tf.stack([labels] * nrof_rows)
        labels_matrix_t = tf.transpose(labels_matrix, [1, 0, 2])

        embeddings_diff = tf.subtract(embeddings_matrix, embeddings_matrix_t)
        embeddings_l2 = tf.sqrt(tf.reduce_sum(tf.square(embeddings_diff), axis=1))
        labels_diff = tf.subtract(labels_matrix, labels_matrix_t)

        same_class_idx = tf.where(tf.equal(labels_diff, 0))
        not_same_idx = tf.where(tf.not_equal(labels_diff, 0))

        same_embeddings = tf.gather_nd(embeddings_l2, same_class_idx)
        not_same_embeddings = tf.gather_nd(labels_diff, not_same_idx)

        with tf.variable_scope('verif_loss'):
            pos_dist = tf.reduce_sum(tf.multiply(0.5, tf.square(same_embeddings)))
            neg_dist = tf.reduce_sum(tf.multiply(0.5, tf.maximum(0.0,
                                                            tf.subtract(alpha, tf.square(not_same_embeddings)))))
            loss = pos_dist + neg_dist

        return loss
