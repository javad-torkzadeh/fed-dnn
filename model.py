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
import keras
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from sklearn.utils import shuffle


class Model:
    def __init__(self, name, num_classes, input_shape, activation, loss, optimizer, learning_rate):
        if optimizer == 'sgd':
            optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
        elif optimizer == 'adam':
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        if name == 'mlp':
            model = keras.Sequential()
            model.add(Flatten(input_shape=input_shape))
            model.add(Dense(units=128, activation=activation))
            model.add(Dense(units=64, activation=activation))
            model.add(Dense(units=num_classes, activation='softmax'))
        elif name == 'cnn':
            model = keras.Sequential()
            model.add(Conv2D(64, (5, 5), activation=activation, input_shape=input_shape))
            model.add(MaxPooling2D((2, 2)))
            model.add(Conv2D(32, (5, 5), activation=activation))
            model.add(MaxPooling2D((2, 2)))
            model.add(Dense(units=32, activation=activation))
            model.add(Dense(units=num_classes, activation='softmax'))
        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
        self.model = model

    def train(self, batch_size, train_x, train_y):
        batch_num = np.ceil(len(train_x) / batch_size)

        train_x, train_y = shuffle(train_x, train_y)
        batches = np.array_split(train_x, batch_num)
        labels = np.array_split(train_y, batch_num)
        for (x, y) in zip(batches, labels):
            self.model.train_on_batch(x, y)

    def evaluate(self, test_x, test_y):
        return self.model.evaluate(test_x, test_y)
