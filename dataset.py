import os.path
import numpy as np
import random
import cv2

class Dataset:
    _dataset_dirs = []
    _training_paths = []

    def __init__(self, dataset_dirs):
        self._dataset_dirs = dataset_dirs

    def read_dataset(self):
        print('Get classes and images path.')
        classes_path = self._get_path()

        class_names = list(classes_path.keys())
        class_count = len(class_names)

        print('Creating training_paths, training_labels, validation_paths, validation_labels.')
        self._training_paths, training_labels = self._get_training_set(class_count, classes_path, class_names)

        return self._training_paths, training_labels, class_count

    def get_mean_px_train(self):
        r = 0.
        g = 0.
        b = 0.
        if len(self._training_paths) > 1000:
            img_count = 1000
        else:
            img_count = len(self._training_paths)

        for idx in range(img_count):
            img = cv2.imread(self._training_paths[idx])

            count = img.shape[0] * img.shape[1]
            t = len(img[:, :, 0])
            b += np.sum(img[:, :, 0]) / count
            g += np.sum(img[:, :, 1]) / count
            r += np.sum(img[:, :, 2]) / count

        r_mean = np.float32(r) / img_count
        g_mean = np.float32(g) / img_count
        b_mean = np.float32(b) / img_count

        return r_mean, g_mean, b_mean

    def _get_training_set(self, class_count, classes_path, class_names):
        training_paths = []
        training_labels = []
        validation_paths = {}

        for i in range(class_count):
            image_paths = classes_path[class_names[i]]

            training_len = len(image_paths)
            training_image_paths = image_paths[:training_len]

            training_paths += training_image_paths
            training_labels += [i] * training_len

        return training_paths, training_labels


    # get path from dir to define classes path where are the images
    def _get_path(self):
        paths = {}
        for dataset_dir in self._dataset_dirs.split(':'):
            classes = os.listdir(dataset_dir)
            nrof_classes = len(classes)

            for i in range(nrof_classes):
                class_name = classes[i]
                facedir = os.path.join(dataset_dir, class_name)

                if os.path.isdir(facedir):
                    images = os.listdir(facedir)
                    image_paths = [os.path.join(facedir, img) for img in images][0:10]

                    if class_name in paths:
                        total_image_paths = paths[class_name]
                        total_image_paths += image_paths
                        paths[class_name] = total_image_paths
                    else:
                        paths[class_name] = image_paths

        return paths

    def read_lfw_dataset(self, lfw_dir, pairs_filename, file_ext):
        pairs = []
        with open(pairs_filename, 'r') as f:
            for line in f.readlines()[1:]:
                pair = line.strip().split()
                pairs.append(pair)

        nrof_skipped_pairs = 0
        path_list = []
        issame_list = []
        for pair in pairs:
            if len(pair) == 3:
                path0 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1]) + '.' + file_ext)
                path1 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2]) + '.' + file_ext)
                issame = True
            elif len(pair) == 4:
                path0 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1]) + '.' + file_ext)
                path1 = os.path.join(lfw_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3]) + '.' + file_ext)
                issame = False

            if os.path.exists(path0) and os.path.exists(path1):  # Only add the pair if both paths exist
                path_list += (path0, path1)
                issame_list.append(issame)
            else:
                nrof_skipped_pairs += 1

        if nrof_skipped_pairs > 0:
            print('Skipped %d image pairs' % nrof_skipped_pairs)

        return path_list, issame_list
