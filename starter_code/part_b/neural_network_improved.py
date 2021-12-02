import matplotlib.pyplot as plt

from starter_code.utils import *
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import numpy as np
import torch


def load_data(base_path="../data"):
    """ Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data


class AutoEncoder(nn.Module):
    def __init__(self, num_question, k1, k2):
        """ Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(num_question, k1)
        self.s = nn.Linear(k1, k2)
        self.t = nn.Linear(k2, k1)
        self.h = nn.Linear(k1, num_question)

    def get_weight_norm(self):
        """ Return ||W^1||^2 + ||W^2||^2.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2) ** 2
        s_w_norm = torch.norm(self.s.weight, 2) ** 2
        t_w_norm = torch.norm(self.t.weight, 2) ** 2
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        return g_w_norm + s_w_norm + t_w_norm + h_w_norm

    def forward(self, inputs):
        """ Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        #####################################################################
        # TODO:                                                             #
        # Implement the function as described in the docstring.             #
        # Use sigmoid activations for f and g.                              #
        #####################################################################
        f1 = self.g(inputs)
        encode = F.sigmoid(f1)
        f2 = self.s(encode)
        mid = F.sigmoid(f2)
        f3 = self.t(mid)
        third = F.sigmoid(f3)
        f4 = self.h(third)
        out = F.sigmoid(f4)
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
        return out


def train(model, lr, train_data, zero_train_data, valid_data, num_epoch, lamb=0):
    """ Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: None
    """
    # TODO: Add a regularizer to the cost function.

    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]

    train_list = []
    val_list = []

    for epoch in range(0, num_epoch):
        train_loss = 0.

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            target[0][nan_mask] = output[0][nan_mask]

            loss = torch.sum((output - target) ** 2.) + model.get_weight_norm() * lamb * 0.5
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

        valid_acc = evaluate(model, zero_train_data, valid_data)
        # print("Epoch: {} \tTraining Cost: {:.6f}\t "
        #       "Valid Acc: {}".format(epoch, train_loss, valid_acc))

        train_list.append(train_loss)
        val_list.append(valid_acc)
    return train_list, val_list
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def evaluate(model, train_data, valid_data):
    """ Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)


def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()

    #####################################################################
    # TODO:                                                             #
    # Try out 5 different k and select the best k using the             #
    # validation set.                                                   #
    #####################################################################
    # Set model hyperparameters.
    k_list = [10,50,100,200,500]
    rev_k_list = [500,200,100,50,10]
    num_student = train_matrix.shape[0]
    num_question = train_matrix.shape[1]

    # Set optimization hyperparameters.
    lr_list = [0.01, 0.05, 0.1, 0.15]
    num_epoch_list = [10,25,50]
    lamb_list = [0.001, 0.01, 0.1, 1]
    # for i, k1 in enumerate(rev_k_list):
    #     for j, k2 in enumerate(rev_k_list[i:]):
    #         for lr in lr_list:
    #             for epoch in num_epoch_list:
    #                 model = AutoEncoder(num_question, k1, k2)
    #                 train_list, val_list = train(model, lr, train_matrix, zero_train_matrix,
    #                                              valid_data, epoch, lamb=0)
    #                 print(f"k1 is {k1}, k2 is {k2}, lr is {lr}, num_epoch is {epoch}, best accuracy rate is {max(val_list)} when epoch is {val_list.index(max(val_list))}")

    # for lamb in lamb_list:
    #     train_list2, val_list2 = train(model, best_lr, train_matrix, zero_train_matrix, valid_data, best_num_epoch,
    #                                    lamb=lamb)
    #     test_acc = evaluate(model, zero_train_matrix, test_data)
    #     print(f"for lamb = {lamb}, best accuracy rate for validation is {max(val_list2)}, test accuracy is {test_acc}")
    #

    best_lr = 0.05
    best_num_epoch = 50
    k1 = 100
    k2 = 10
    best_lamb = 0.001

    model = AutoEncoder(num_question, k1, k2)
    train_list1, val_list1 = train(model, best_lr, train_matrix, zero_train_matrix, valid_data, best_num_epoch, lamb=best_lamb)
    test_acc = evaluate(model, zero_train_matrix, test_data)
    plt.figure(1)
    plt.plot(list(range(best_num_epoch)), train_list1, label="training lost")
    plt.xlabel("epoch")
    plt.ylabel("training loss")
    plt.title("epoch vs training loss")
    plt.figure(2)
    plt.plot(list(range(best_num_epoch)), val_list1, label="training lost")
    plt.xlabel("epoch")
    plt.ylabel("validation accuracy")
    plt.title("epoch vs validation accuracy")
    plt.show()
    print(f"test accuracy is {test_acc}")





    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
