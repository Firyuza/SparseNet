import numpy as np


class NeuralCorrelation:
    _S = 1 / 256
    _lambda = 0.75
    _dropping_matrix = []
    _std_deviation_current = []
    _mean_matrix_current = []
    _std_deviation_previous = []
    _mean_matrix_previous = []

    def __init__(self, S=1 / 256, lambda_=0.75):
        self._S = S
        self._lambda = lambda_

    def sparsify_layer(self, current_layer, previous_layer, filter_size, sqr_matrix_previous=[],
                       mean_matrix_previous=[], sqr_matrix_current=[], mean_matrix_current=[], ):
        self.current_layer = current_layer
        self.previous_layer = previous_layer

        self._std_deviation_current = sqr_matrix_previous
        self._mean_matrix_current = mean_matrix_previous
        self._std_deviation_previous = sqr_matrix_current
        self._mean_matrix_previous = mean_matrix_current

        self._S = np.float(1. / self.previous_layer.shape[3])

        # set size for dropping matrix
        self.filter_size = filter_size
        self._set_dropping_matrix_size()

    def dropping_matrix_fc(self):
        shape = self.previous_layer.shape
        dim = 1
        for d in shape[1:]:
            dim *= d
        previous_layer_list = np.reshape(self.previous_layer, [-1, dim])
        mean_matrix_previous = np.reshape(self._mean_matrix_previous, [-1, dim])
        std_deviation_previous = np.reshape(self._std_deviation_previous, [-1, dim])

        correlation_pos = dict()
        correlation_neg = dict()
        depth = 0

        for cur_idx in xrange(self.current_layer.shape[1]):
            idx = 0
            for prev_idx in xrange(previous_layer_list.shape[1]):
                # print cur_idx
                # print self.current_layer[0][cur_idx]
                # print self._mean_matrix_current[0][cur_idx]
                # print self._std_deviation_current[0][cur_idx]
                # print prev_idx
                # print previous_layer_list[0][prev_idx]
                # print mean_matrix_previous[0][prev_idx]
                # print std_deviation_previous[0][prev_idx]
                # print '==========================='

                # correlation coefficient
                r = (np.float((self.current_layer[0][cur_idx] - self._mean_matrix_current[0][cur_idx])) * np.float((
                    previous_layer_list[0][prev_idx] - mean_matrix_previous[0][prev_idx]))) * 1. / (
                        self._std_deviation_current[0][cur_idx] * std_deviation_previous[0][prev_idx])

                filter_idx = [idx, depth]
                if r > 0:
                    correlation_pos[idx] = (r, filter_idx)
                else:
                    correlation_neg[idx] = (abs(r), filter_idx)
                idx += 1

            depth += 1

        # rank positive coefficients and then store lSK+ from first half and (1-l)SK+ from second half, others remove
        self._rank_pos_correlation(correlation_pos)

        # rank negative coefficients and then store lSK- from first half and (1-l)SK- from second half, others remove
        self._rank_neg_correlation(correlation_neg)

        return self._dropping_matrix

    def dropping_matrix_lc(self):
        idx = 0
        correlation_pos = dict()
        correlation_neg = dict()

        for cur_depth in range(self.current_layer.shape[3]):
            filter_x = 0
            for row_idx in range(self.current_layer.shape[1]):
                for column_idx in range(self.current_layer.shape[2]):
                    filter_y = 0
                    for z in range(self.filter_size[2]):
                        for x in range(3):
                            for y in range(3):
                                r = ((self.current_layer[0][row_idx][column_idx][cur_depth] -
                                      self._mean_matrix_current[0][row_idx][column_idx][cur_depth]) *
                                     (self.previous_layer[0][x + row_idx][y + column_idx][z] -
                                      self._mean_matrix_previous[0][x + row_idx][y + column_idx][z])) / (
                                        self._std_deviation_current[0][row_idx][column_idx][cur_depth] *
                                        self._std_deviation_previous[0][x + row_idx][y + column_idx][z])

                                filter_idx = [filter_x, filter_y, cur_depth]
                                if r > 0:
                                    correlation_pos[idx] = (r, filter_idx)
                                else:
                                    correlation_neg[idx] = (abs(r), filter_idx)
                                idx += 1
                                filter_y += 1
                    filter_x += 1

        # rank positive coefficients and then store lSK+ from first half and (1-l)SK+ from second half, others remove
        self._rank_pos_correlation(correlation_pos)

        # rank negative coefficients and then store lSK- from first half and (1-l)SK- from second half, others remove
        self._rank_neg_correlation(correlation_neg)

        return self._dropping_matrix

    def dropping_matrix_conv(self):
        correlation_pos = dict()
        correlation_neg = dict()
        idx = 0
        prev_width = self.previous_layer.shape[1]
        prev_height = self.previous_layer.shape[2]

        for feature_map_idx in range(self.filter_size[3]):
            for k_depth in range(self.filter_size[2]):
                for k_height in range(self.filter_size[0]):
                    for k_width in range(self.filter_size[1]):
                        r = 0
                        for a_row in range(self.current_layer.shape[1]):
                            for a_column in range(self.current_layer.shape[2]):
                                # correlation coefficient
                                if (0 <= a_row + k_height - 1 < prev_width) and (
                                        0 <= a_column + k_width - 1 < prev_height):
                                    r_part = ((self.current_layer[0][a_row][a_column][feature_map_idx] -
                                               self._mean_matrix_current[0][a_row][a_column][feature_map_idx]) *
                                              (self.previous_layer[0][a_row + k_height - 1][a_column + k_width - 1][k_depth] -
                                               self._mean_matrix_previous[0][a_row + k_height - 1][a_column + k_width - 1][
                                                   k_depth])) / (
                                                 self._std_deviation_current[0][a_row][a_column][feature_map_idx] *
                                                 self._std_deviation_previous[0][a_row + k_height - 1][a_column + k_width - 1][
                                                     k_depth])
                                    r += r_part

                        filter_idx = [k_width, k_height, k_depth, feature_map_idx]
                        if r > 0:
                            correlation_pos[idx] = (r, filter_idx)
                        else:
                            correlation_neg[idx] = (abs(r), filter_idx)
                        idx += 1

        # rank positive coefficients and then store lSK+ from first half and (1-l)SK+ from second half, others remove
        self._rank_pos_correlation(correlation_pos)

        # rank negative coefficients and then store lSK- from first half and (1-l)SK- from second half, others remove
        self._rank_neg_correlation(correlation_neg)

        return self._dropping_matrix

    def _rank_pos_correlation(self, correlations):
        count = 0
        first_half_count = self._lambda * self._S * len(correlations)
        second_half_count = (1 - self._lambda) * self._S * len(correlations)
        half = len(correlations) // 2
        sorted_correlations = [(k, correlations[k]) for k in sorted(correlations, key=correlations.get, reverse=True)]
        for key, value in sorted_correlations:
            if count <= first_half_count or (half < count < half + second_half_count):
                self._set_value_to_dropping_matrix(value)
            else:
                return

            count += 1

    def _rank_neg_correlation(self, correlations):
        count = 0
        part = self._lambda * self._S * len(correlations)
        sorted_correlations = [(k, correlations[k]) for k in sorted(correlations, key=correlations.get, reverse=True)]
        for key, value in sorted_correlations:
            if count <= part:
                self._set_value_to_dropping_matrix(value)
            else:
                return

            count += 1

    def _set_value_to_dropping_matrix(self, value):
        if len(value[1]) == 2:
            self._dropping_matrix[value[1][0]][value[1][1]] = 1
        elif len(value[1]) == 3:
            self._dropping_matrix[value[1][0]][value[1][1]][value[1][2]] = 1
        elif len(value[1]) == 4:
            self._dropping_matrix[value[1][0]][value[1][1]][value[1][2]][value[1][3]] = 1

    def _set_dropping_matrix_size(self):
        if len(self.filter_size) == 2:
            self._dropping_matrix = np.zeros((self.filter_size[0], self.filter_size[1]), dtype=np.float32)
        elif len(self.filter_size) == 3:
            self._dropping_matrix = np.zeros((self.filter_size[0], self.filter_size[1], self.filter_size[2]), dtype=np.float32)
        elif len(self.filter_size) == 4:
            self._dropping_matrix = np.zeros((self.filter_size[0], self.filter_size[1], self.filter_size[2], self.filter_size[3]), dtype=np.float32)
