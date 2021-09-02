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

import argparse
from dataset import Dataset
from model import Model
import numpy as np


def main():
    """parse input arguments"""
    parser = argparse.ArgumentParser(description='',
                                     usage=f"python simulate.py ")

    # arguments
    parser.add_argument("--approach_name", "--approach", type=str, help="approach_name", default='centralized')

    parser.add_argument("--dataset_name", "--dataset", type=str, help="dataset name", default='mnist')

    parser.add_argument("--model_name", "--model", type=str, help="Model to be trained", default='mlp')

    parser.add_argument("--batch_size", "--batch-size", type=int, help="batch size", default=128)

    parser.add_argument("--learning_rate", "--learning-rate", type=float, help="learning rate", default=0.01)

    parser.add_argument("--loss_function_name", "--loss", type=str, help="loss function name",
                        default='categorical_crossentropy')

    parser.add_argument("--optimizer_name", "--optimizer", type=str, help="optimizer name", default='sgd')

    parser.add_argument("--activation_name", "--activation", type=str, help="activation function name", default=None)

    parser.add_argument("--epoch", "--epoch", type=int, help="epoch_value", default=10)

    parser.add_argument("--fed_num", "--fed", type=int, help="number_of_fed", default=10)

    parser.add_argument("--distribution", "--distribution", type=str, help="type_of_distribution", default='balance')

    parser.add_argument("--local_epoch", "--local-epoch", type=int, help="number_of_local_epoch", default=1)

    args = parser.parse_args()

    if args.approach_name == 'centralized':
        file_name = f"{args.approach_name}-{args.dataset_name}-{args.model_name}-{args.activation_name}-{args.loss_function_name}-{args.optimizer_name}-{args.learning_rate}-{args.batch_size}.csv"

    elif args.approach_name == 'federated':
        file_name = f"{args.approach_name}-{args.dataset_name}-{args.model_name}-{args.activation_name}-{args.loss_function_name}-{args.optimizer_name}-{args.learning_rate}-{args.batch_size}-{args.distribution}.csv"

    file = open(file_name, "w")
    file.write("epoch,loss,accuracy\n")

    dataset = Dataset(args.dataset_name)
    dataset.normalize()
    dataset.one_hot()

    if args.approach_name == "centralized":
        centralized_model = Model(args.model_name, dataset.num_classes, dataset.input_shape, args.activation_name,
                                  args.loss_function_name, args.optimizer_name, args.learning_rate)

        for i in range(1, args.epoch + 1):
            centralized_model.centralized_train(args.batch_size, dataset.train_x, dataset.train_y)
            loss, accuracy = centralized_model.evaluate(dataset.test_x, dataset.test_y)
            file.write(f"{i},{loss},{accuracy}\n")
        file.close()

    elif args.approach_name == "federated":
        fed_train_x, fed_train_y = dataset.split(args.fed_num, args.distribution)
        fed_models = [Model(args.model_name, dataset.num_classes, dataset.input_shape, args.activation_name,
                            args.loss_function_name, args.optimizer_name, args.learning_rate) for _ in range(args.fed_num)]
        model_test = Model(args.model_name, dataset.num_classes, dataset.input_shape, args.activation_name,
                           args.loss_function_name, args.optimizer_name, args.learning_rate)
        global_weights = fed_models[0].model.get_weights()
        local_weights = []
        local_samples = []

        for i in range(1, args.epoch + 1):
            for j in range(args.fed_num):
                fed_models[j].model.set_weights(global_weights)
                fed_models[j].train_fedavg(args.batch_size, fed_train_x[j], fed_train_y[j], args.local_epoch)
                local_weights.append(fed_models[j].model.get_weights())
                local_samples.append(len(fed_train_x[j]))
            global_weights = (np.dot(np.array(local_samples), np.array(local_weights))) / np.sum(local_samples)
            local_weights = []
            local_samples = []
            model_test.model.set_weights(global_weights)
            loss, accuracy = model_test.evaluate(dataset.test_x, dataset.test_y)
            file.write(f"{i},{loss},{accuracy}\n")

        # for i in range(1, args.epoch + 1):
        #     model.train(args.batch_size, dataset.train_x, dataset.train_y)
        #     loss, accuracy = model.evaluate(dataset.test_x, dataset.test_y)
        #     file.write(f"{i},{loss},{accuracy}\n")
        #     print(f"epoch = {i}")

        file.close()



if __name__ == "__main__":
    main()
