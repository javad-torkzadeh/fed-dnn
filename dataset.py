"""

    Copyright 2021 Javad TorkzadehMahani. All Rights Reserved.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""

import numpy as np
from keras.datasets import cifar10, mnist
from sklearn.utils import shuffle


class Dataset:
    def __init__(self, name):
        if name == 'cifar10':
            (train_x, train_y), (test_x, test_y) = cifar10.load_data()
        if name == 'mnist':
            (train_x, train_y), (test_x, test_y) = mnist.load_data()
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.num_classes = len(np.unique(train_y))
        self.input_shape = train_x.shape[1:]

    def split(self, n_client, split_type):
        x_train_splits = []
        y_train_splits = []
        if split_type == 'balance':
            x_train_splits = np.array_split(self.train_x, n_client)
            y_train_splits = np.array_split(self.train_y, n_client)
            return x_train_splits, y_train_splits

        elif split_type == 'imbalance':
            x_split = np.array_split(self.train_x, np.array([1000, 3000, 6000, 10000, 15000, 21000, 28000, 36000, 45000]))
            y_split = np.array_split(self.train_y, np.array([1000, 3000, 6000, 10000, 15000, 21000, 28000, 36000, 45000]))

            # counter = 0
            #  ax = []
            #  ay = []
            #  bx = []
            #  by = []
            #  cx = np.array_split(self.train_x, 50)
            #  cy = np.array_split(self.train_y, 50)
            #  for i in range(50):
            #      for j in range(counter, counter + i):
            #          ax.extend(cx[j])
            #          ay.extend(cy[j])
            #          counter = counter + 1
            #      bx.append(ax)
            #      by.append(ay)
            #      return bx, by

            return x_split, y_split

        elif split_type == 'non-iid':
            index_list = [[], [], [], [], [], [], [], [], [], []]
            for index in range(len(self.train_y)):
                label = np.argmax(self.train_y[index])
                index_list[label].append(index)

            classify = shuffle(index_list)
            two_D_classification = []
            for b in classify:
                two_D_classification.append(np.array_split(b, 2))

            non_iid_indexes = []
            non_iid_indexes.append(np.concatenate((two_D_classification[1][0], two_D_classification[7][1])))
            non_iid_indexes.append(np.concatenate((two_D_classification[1][1], two_D_classification[2][1])))
            non_iid_indexes.append(np.concatenate((two_D_classification[5][0], two_D_classification[7][0])))
            non_iid_indexes.append(np.concatenate((two_D_classification[5][1], two_D_classification[4][0])))
            non_iid_indexes.append(np.concatenate((two_D_classification[6][0], two_D_classification[2][0])))
            non_iid_indexes.append(np.concatenate((two_D_classification[6][1], two_D_classification[4][1])))
            non_iid_indexes.append(np.concatenate((two_D_classification[3][0], two_D_classification[8][0])))
            non_iid_indexes.append(np.concatenate((two_D_classification[8][1], two_D_classification[9][0])))
            non_iid_indexes.append(np.concatenate((two_D_classification[0][0], two_D_classification[3][1])))
            non_iid_indexes.append(np.concatenate((two_D_classification[9][1], two_D_classification[0][1])))
            non_iid_x_split = []
            non_iid_y_split = []
            # for i in range(10):
            #      values_of_i_index = []
            #     for j in range(len(non_iid_indexes[i])):
            #        values_of_i_index.extend(non_iid_indexes[i][j])
            #    non_iid_value_split.append(values_of_i_index[i])

            for k in range(10):
                non_iid_x_split.append(self.train_x[non_iid_indexes[k]])
                non_iid_y_split.append(self.train_y[non_iid_indexes[k]])

            return non_iid_x_split, non_iid_y_split

        # count = 0
        #
        # for i in shuffle(range(2)):
        #     k = shuffle(range(10))
        #     for j in k:
        #         d[count].append(np.concatenate((self.train_x[two_D_classification[k][i]], self.train_x[two_D_classification[k-count][i]])))
        #         count = count + 1
        #         if count == 5 and count == 10:
        #             break
        #
        # return d

    def normalize(self):
        self.train_x = self.train_x / 255
        self.test_x = self.test_x / 255

    def one_hot(self):
        self.train_y = np.eye(self.num_classes)[self.train_y.flatten()]
        self.test_y = np.eye(self.num_classes)[self.test_y.flatten()]

    def get_train_set(self):
        return self.train_x, self.train_y

    def get_test_set(self):
        return self.test_x, self.train_y
