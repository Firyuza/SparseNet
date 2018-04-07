import numpy as np

def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc

def calculate_roc(first_persons, second_persons, actual_issame, thresholds, nrof_folds=10):
    actual_issame_array = np.array(actual_issame)

    nrof_thresholds = len(thresholds)

    true_indices = np.where(True == actual_issame_array)[0]
    false_indices = np.where(False == actual_issame_array)[0]

    folds_true_len = len(true_indices) // nrof_folds
    folds_false_len = len(false_indices) // nrof_folds

    accuracy = np.zeros(shape=nrof_folds, dtype=np.float32)

    diff = np.subtract(first_persons, second_persons)
    dist = np.sum(np.square(diff), 1)

    best_threshold_indices = np.zeros(shape=nrof_folds, dtype=np.float32)
    for fold_idx in range(nrof_folds):
        start_idx = fold_idx * folds_true_len
        test_set_true_idx = true_indices[start_idx:start_idx + folds_true_len]
        train_set_true_idx = np.concatenate((true_indices[:start_idx], true_indices[start_idx + folds_true_len:]))
        start_idx = fold_idx * folds_false_len
        test_set_false_idx = false_indices[start_idx:start_idx + folds_false_len]
        train_set_false_idx = np.concatenate((false_indices[:start_idx], false_indices[start_idx + folds_false_len:]))

        test_set_issame = np.concatenate((actual_issame_array[test_set_true_idx], actual_issame_array[test_set_false_idx]))
        test_set = np.concatenate((dist[test_set_true_idx], dist[test_set_false_idx]))
        # test_set_issame = np.zeros(len(test_set_true_idx) + len(test_set_false_idx))
        # test_set_issame[:len(test_set_true_idx)] = actual_issame[test_set_true_idx]
        # test_set_issame[len(test_set_true_idx):] = actual_issame[test_set_false_idx]

        train_set = np.concatenate((dist[train_set_true_idx], dist[train_set_false_idx]))
        train_set_issame = np.concatenate((actual_issame_array[train_set_true_idx], actual_issame_array[train_set_false_idx]))
        # train_set_issame = np.zeros(len(train_set_true_idx) + len(train_set_false_idx))
        # train_set_issame[:len(train_set_true_idx)] = actual_issame[train_set_true_idx]
        # train_set_issame[len(train_set_true_idx):] = actual_issame[train_set_false_idx]

        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, train_set, train_set_issame)

        best_threshold_indices[fold_idx] = np.argmax(acc_train)
        _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_indices[fold_idx]],
                                                      test_set,
                                                      test_set_issame)

    mean_acc = np.mean(accuracy)

    return mean_acc