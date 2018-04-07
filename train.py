from datetime import datetime
from dataset import Dataset
from batch import Batch
from loss import Loss
from network_architecture.sparsified_nn import SparsifiedNN
from network_architecture.neural_correlation import NeuralCorrelation

import os.path
import argparse
import sys
import tensorflow as tf
import numpy as np
import lfw


def get_new_model_dir(models_dir):
    nowdatetime = str(datetime.now()).replace('-', '_')
    nowdatetime = nowdatetime.replace(':', '_')
    nowdatetime = nowdatetime.replace(' ', '_')
    return models_dir + '/' + nowdatetime


def classification_accuracy(predictions, labels):
    return np.sum(np.argmax(predictions) == np.argmax(labels)) / len(labels)


def main(args):
    subdirs = list(filter(lambda x: 'model' in x, os.listdir(args.models_dir)))
    if len(subdirs) > 0:
        subdirs = filter(lambda x: os.path.join(args.models_dir, x), subdirs)
        latest_subdir = max(subdirs, key=os.path.getmtime)
        if len(os.listdir(latest_subdir)) > 0:
            model_dir = latest_subdir
        else:
            model_dir = get_new_model_dir(args.models_dir)
    else:
        model_dir = get_new_model_dir(args.models_dir)

    os.mkdir(model_dir + '_' + 'info')

    checkpoint_path = os.path.join(args.models_dir, 'model.ckpt')

    dataset = Dataset(args.dataset_dir)
    training_paths, training_labels, num_labels = dataset.read_dataset()
    r_mean, g_mean, b_mean = dataset.get_mean_px_train()

    lfw_pairs, lfw_is_same = dataset.read_lfw_dataset(args.lfw_dir, args.lfw_pairs, 'png')

    sparsified_nn = SparsifiedNN(3, args.image_height, args.image_width, num_of_classes=num_labels)

    layers_to_sparsify = dict()
    weights = sparsified_nn.get_weights('fc')
    layers_to_sparsify[0] = ('non', [])
    layers_to_sparsify[1] = ('fc', weights.get_shape().as_list())
    weights = sparsified_nn.get_weights('lc5b')
    layers_to_sparsify[2] = ('lc5b', weights.get_shape().as_list())
    weights = sparsified_nn.get_weights('lc5a')
    layers_to_sparsify[3] = ('lc5a', weights.get_shape().as_list())
    weights = sparsified_nn.get_weights('conv4b')
    layers_to_sparsify[4] = ('conv4b', weights.get_shape().as_list())
    weights = sparsified_nn.get_weights('conv4a')
    layers_to_sparsify[5] = ('conv4a', weights.get_shape().as_list())
    weights = sparsified_nn.get_weights('conv3b')
    layers_to_sparsify[6] = ('conv3b', weights.get_shape().as_list())
    weights = sparsified_nn.get_weights('conv3a')
    layers_to_sparsify[7] = ('conv3a', weights.get_shape().as_list())
    weights = sparsified_nn.get_weights('conv2b')
    layers_to_sparsify[8] = ('conv2b', weights.get_shape().as_list())
    weights = sparsified_nn.get_weights('conv2a')
    layers_to_sparsify[9] = ('conv2a', weights.get_shape().as_list())
    weights = sparsified_nn.get_weights('conv1b')
    layers_to_sparsify[10] = ('conv1b', weights.get_shape().as_list())
    weights = sparsified_nn.get_weights('conv1a')
    layers_to_sparsify[11] = ('conv1a', weights.get_shape().as_list())

    layers_len = len(layers_to_sparsify)

    # keep dropping matrices for re-training
    dropping_matrices = dict()

    current_layer_from_dir = os.path.join(args.layers_data_dir, 'current_layer')
    if not os.path.exists(current_layer_from_dir):
        os.makedirs(current_layer_from_dir)
    previous_layer_from_dir = os.path.join(args.layers_data_dir, 'previous_layer')
    if not os.path.exists(previous_layer_from_dir):
        os.makedirs(previous_layer_from_dir)

    # loop for sparsifying nn
    for key, value in layers_to_sparsify.items():
        if key > 0:

            print("Sparsify " + value[0] + " layer")

            nc = NeuralCorrelation()

            current_layer_after_training = np.load(current_layer_from_dir + '/current_layer.npy')
            current_layer_after_training_std = np.load(current_layer_from_dir + '/std_current_layer.npy')
            current_layer_after_training_mean = np.load(current_layer_from_dir + '/mean_current_layer.npy')

            previous_layer_after_training = np.load(previous_layer_from_dir + '/previous_layer.npy')
            previous_layer_after_training_std = np.load(previous_layer_from_dir + '/std_previous_layer.npy')
            previous_layer_after_training_mean = np.load(previous_layer_from_dir + '/mean_previous_layer.npy')

            nc.sparsify_layer(current_layer_after_training, previous_layer_after_training, value[1],
                              current_layer_after_training_std,
                              current_layer_after_training_mean,
                              previous_layer_after_training_std,
                              previous_layer_after_training_mean)

            if key > 3:
                dropping_matrix = nc.dropping_matrix_conv()
            elif key == 1:
                dropping_matrix = nc.dropping_matrix_fc()
            else:
                dropping_matrix = nc.dropping_matrix_lc()
            dropping_matrices[key] = (value[0], dropping_matrix)

            sparsified_nn = SparsifiedNN(3, args.image_height, args.image_width, num_labels, dropping_matrices,
                                         value[0])

        graph = sparsified_nn.get_graph()
        print('Building computational graph.')
        with graph.as_default():
            # Training tensorflow part
            train_batch = Batch(training_paths, args.image_height,
                                args.image_width, args.height_offset,
                                args.width_offset, args.batch_size, num_labels, labels=training_labels)
            image_batch, label_batch = train_batch.read_data(args.nrof_preprocess_threads, seed=args.seed,
                                                             phase_train=True, label_smoothing=args.label_smoothing,
                                                             r_mean=r_mean, g_mean=g_mean, b_mean=b_mean)

            tf_learning_rate = tf.placeholder(dtype=tf.float32, shape=None, name='learning_rate')
            global_step = tf.Variable(0, trainable=False, name='global_step')

            prelogits1, prelogits2, logits = sparsified_nn.forward_pass(image_batch, reuse=False)

            verif_loss1 = Loss.get_verif_loss(prelogits1, label_batch, args.verif_loss_alpha)
            verif_loss2 = Loss.get_verif_loss(prelogits2, label_batch, args.verif_loss_alpha)
            batch_prediction = tf.nn.softmax(logits)
            softmax_log_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label_batch))
            regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            loss = softmax_log_loss + verif_loss1 + verif_loss2 + tf.reduce_sum(regularization_losses)
            optimizer = tf.train.RMSPropOptimizer(tf_learning_rate).minimize(loss, global_step)

            # get layers to sparse nn
            if key < (layers_len - 2):
                previous_layer = sparsified_nn.get_layer(layers_to_sparsify.get(key + 2)[0], is_previous=True)
                current_layer = sparsified_nn.get_layer(layers_to_sparsify.get(key + 1)[0])

            # Validation tensorflow part
            lfw_data_batch = Batch(lfw_pairs, args.image_height, args.image_width,
                                   args.height_offset, args.width_offset,
                                   args.validation_batch_size, num_labels, labels=[])
            lfw_batch = lfw_data_batch.read_data(args.nrof_preprocess_threads, seed=args.seed,
                                                 phase_train=False, label_smoothing=0.0)
            _, lfw_prelogits, _ = sparsified_nn.forward_pass(lfw_batch, reuse=True)

        # Training
        with tf.Session(graph=graph) as sess:
            print('Creating a session.')

            # Create a saver
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=3)
            if os.path.exists(checkpoint_path):
                saver.restore(sess, checkpoint_path)

            tf.global_variables_initializer().run()
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            lr = args.learning_rate
            gl_st = sess.run(global_step)
            if lr > 24000:
                lr /= 100.
            elif lr > 16000.:
                lr /= 10.

            # for mean and standard deviation
            sqr_matrix_previous = np.zeros(previous_layer.get_shape().as_list())
            mean_matrix_previous = np.zeros(previous_layer.get_shape().as_list())
            sqr_matrix_current = np.zeros(current_layer.get_shape().as_list())
            mean_matrix_current = np.zeros(current_layer.get_shape().as_list())

            nrof_epochs = args.max_nrof_epochs
            epoch_size = args.epoch_size
            if key == 0:
                nrof_epochs = args.max_nrof_epochs_base
                epoch_size = args.epoch_size_base
            start_epoch = gl_st // epoch_size
            start_epoch_step = gl_st - start_epoch * epoch_size
            count = epoch_size * nrof_epochs

            print('Starting training.')
            for i in range(start_epoch, nrof_epochs):
                for j in range(start_epoch_step, epoch_size):
                    a = datetime.now()
                    step = i * args.epoch_size + j
                    if step == 16000 or step == 240000:
                        lr /= 10.
                    feed_dict = {tf_learning_rate: lr}
                    _, batch_prediction_, label_batch_, previous_layer_, current_layer_ = sess.run(
                        [optimizer, batch_prediction, label_batch, previous_layer, current_layer], feed_dict=feed_dict)
                    batch_accuracy = classification_accuracy(batch_prediction_, label_batch_)
                    b = datetime.now()

                    sqr_matrix_current = sqr_matrix_current + current_layer_ * current_layer_
                    mean_matrix_current = mean_matrix_current + current_layer_
                    sqr_matrix_previous = sqr_matrix_previous + previous_layer_ * previous_layer_
                    mean_matrix_previous = mean_matrix_previous + previous_layer_

                    print('Epoch:', i, 'batch:', j, 'accuracy:', str(batch_accuracy) + '%', 'time:', b - a, 'seconds.')

                # Validation
                def run_pairs_forward_pass(prelogits):
                    nrof_images = len(lfw_pairs)
                    for j in range(nrof_images // args.validation_batch_size):
                        a = datetime.now()
                        prelogits_ = sess.run(prelogits)
                        if j == 0:
                            prelogits_length = prelogits_.shape[1]
                            all_validation_prelogits = np.zeros((nrof_images, prelogits_length))
                        begin = j * args.validation_batch_size
                        end = begin + args.validation_batch_size
                        all_validation_prelogits[begin:end] = prelogits_
                        b = datetime.now()
                        print('Forward pass from', begin, 'to', end, 'images time:', b - a, 'seconds.')
                    first_persons = all_validation_prelogits[0:2]
                    second_persons = all_validation_prelogits[1:2]
                    return first_persons, second_persons

                # Validate on LFW
                print('Validate on LFW with picked threshold. Running forward pass on LFW images.')
                first_persons, second_persons = run_pairs_forward_pass(lfw_prelogits)
                thresholds = np.arange(0, 4, 0.001)
                _, _, lfw_accuracy = lfw.calculate_roc(first_persons, second_persons, lfw_is_same, thresholds)
                print('LFW accuracy:', lfw_accuracy)

            # save layers in file
            mean_matrix_previous = mean_matrix_previous / count
            np.save(previous_layer_from_dir + '/mean_previous_layer', mean_matrix_previous)
            sqr_matrix_previous -= mean_matrix_previous * mean_matrix_previous
            np.save(previous_layer_from_dir + '/std_previous_layer', sqr_matrix_previous)
            np.save(previous_layer_from_dir + '/previous_layer', previous_layer_)

            mean_matrix_current = mean_matrix_current / count
            np.save(current_layer_from_dir + '/mean_current_layer', mean_matrix_current)
            sqr_matrix_current -= mean_matrix_current * mean_matrix_current
            np.save(current_layer_from_dir + '/std_current_layer', sqr_matrix_current)
            np.save(current_layer_from_dir + '/current_layer', current_layer_)

            saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)

            coord.request_stop()
            coord.join(threads)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--models_dir', type=str,
                        help='Directory where to write trained models and checkpoints.', default='models')
    parser.add_argument('--dataset_dir', type=str,
                        help='Path to the data directory containing aligned face patches. Multiple directories are separated with colon.',
                        default='/media/firiuza/WININSTALL/ML/vgg_face_dataset_aligned_182/files/')
    parser.add_argument('--layers_data_dir', type=str,
                        help='Path to the data directory containing layers after traing for sparsifying. Multiple directories are separated with colon.',
                        default='layers_to_sparsify/')
    parser.add_argument('--dropping_matrices', type=str,
                        help='Path to the data directory to store dropping matrices for sparse layers. Multiple directories are separated with colon.',
                        default='dropping_matrices/')
    parser.add_argument('--architecture', type=str,
                        help='Model definition. Points to a module containing the definition of the inference graph.',
                        default='architectures.centrer_loss_architecture')
    parser.add_argument('--max_nrof_epochs_base', type=int,
                        help='Number of epochs to run.', default=140)
    parser.add_argument('--max_nrof_epochs', type=int,
                        help='Number of epochs to run.', default=70)
    parser.add_argument('--batch_size', type=int,
                        help='Number of images to process in a batch.', default=64)
    parser.add_argument('--image_height', type=int,
                        help='Image height in pixels.', default=112)
    parser.add_argument('--image_width', type=int,
                        help='Image width in pixels.', default=96)
    parser.add_argument('--height_offset', type=int,
                        help='Augmentation height offset to make crop from', default=20)
    parser.add_argument('--width_offset', type=int,
                        help='Augmentation width offset to make crop from', default=17)
    parser.add_argument('--epoch_size_base', type=int,
                        help='Number of batches per epoch for base model.', default=1000)
    parser.add_argument('--epoch_size', type=int,
                        help='Number of batches per epoch for sparse models.', default=1000)
    parser.add_argument('--keep_probability', type=float,
                        help='Keep probability of dropout for the fully connected layer(s).', default=0.5)
    parser.add_argument('--weight_decay', type=float,
                        help='L2 weight regularization.', default=0.0001)
    parser.add_argument('--label_smoothing', type=float,
                        help='Label smoothing parameter.', default=0.0001)
    parser.add_argument('--center_loss_factor', type=float,
                        help='Center loss factor.', default=0.003)
    parser.add_argument('--center_loss_alpha', type=float,
                        help='Center update rate for center loss.', default=0.5)
    parser.add_argument('--verif_loss_alpha', type=float,
                        help='Alpha for verification loss.', default=0.5)
    parser.add_argument('--learning_rate', type=float,
                        help='Initial learning rate.', default=0.01)
    parser.add_argument('--seed', type=int,
                        help='Random seed.', default=666)
    parser.add_argument('--nrof_preprocess_threads', type=int,
                        help='Number of preprocessing (data loading and augumentation) threads.', default=4)
    parser.add_argument('--validation_set_persent', type=float,
                        help='Persent of validation dataset taken from the whole dataset.', default=0.1)
    parser.add_argument('--max_validation_set_pairs', type=int,
                        help='Max validation dataset pairs.', default=60)
    parser.add_argument('--validation_batch_size', type=int,
                        help='Batch size for validation.', default=64)
    parser.add_argument('--lfw_pairs', type=str,
                        help='The file containing the pairs to use for validation.', default='./data/pairs.txt')
    parser.add_argument('--lfw_dir', type=str,
                        help='Path to the data directory containing aligned face patches.',
                        default='/media/firiuza/WININSTALL/ML/lfw_aligned')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
