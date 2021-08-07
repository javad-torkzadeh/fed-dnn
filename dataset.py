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
