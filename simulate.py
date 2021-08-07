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


def main():
    """parse input arguments"""
    parser = argparse.ArgumentParser(description='',
                                     usage=f"python simulate.py ")

    # arguments
    parser.add_argument("--dataset_name", "--dataset", type=str, help="dataset name", default='mnist')

    parser.add_argument("--model_name", "--model", type=str, help="Model to be trained", default='mlp')

    parser.add_argument("--batch_size", "--batch-size", type=int, help="batch size", default=128)

    parser.add_argument("--learning_rate", "--learning-rate", type=float, help="learning rate", default=0.01)

    parser.add_argument("--loss_function_name", "--loss", type=str, help="loss function name", default='categorical_crossentropy')

    parser.add_argument("--optimizer_name", "--optimizer", type=str, help="optimizer name", default='sgd')

    parser.add_argument("--activation_name", "--activation", type=str, help="activation function name", default=None)

    parser.add_argument("--epoch", "--epoch", type=int, help="epoch_value", default=5)

    args = parser.parse_args()

    file_name = f"{args.dataset_name}-{args.model_name}-{args.activation_name}-{args.loss_function_name}-{args.optimizer_name}-{args.learning_rate}-{args.batch_size}.csv"

    file = open(file_name, "w")
    file.write("epoch,loss,accuracy\n")

    dataset = Dataset(args.dataset_name)
    model = Model(args.model_name, dataset.num_classes, dataset.input_shape, args.activation_name, args.loss_function_name, args.optimizer_name, args.learning_rate)
    dataset.normalize()
    dataset.one_hot()

    for i in range(1, args.epoch+1):
        model.train(args.batch_size, dataset.train_x, dataset.train_y)
        loss, accuracy = model.evaluate(dataset.test_x, dataset.test_y)
        file.write(f"{i},{loss},{accuracy}\n")
        print(f"epoch = {i}")
    file.close()


if __name__ == "__main__":
    main()
