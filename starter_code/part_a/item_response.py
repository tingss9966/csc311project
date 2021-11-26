import matplotlib.pyplot as plt

from starter_code.utils import *

import numpy as np


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_lklihood = 0.
    for i in range(len(data["user_id"])):
        theta_i = theta[data["user_id"][i]]
        beta_j = beta[data["question_id"][i]]
        sig = sigmoid(theta_i - beta_j)
        log_lklihood += (data['is_correct'][i] * np.log(sig) + (1-data['is_correct'][i]) * np.log(1-sig))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    theta_t = np.zeros(theta.shape)
    beta_t = np.zeros(beta.shape)
    for i in range(len(data["user_id"])):
        theta_i = theta[data["user_id"][i]]
        beta_j = beta[data["question_id"][i]]
        cij = data['is_correct'][i]
        theta_t[data["user_id"][i]] += np.exp(theta_i) / (np.exp(theta_i)+np.exp(beta_j)) - cij
        beta_t[data["question_id"][i]] += cij - np.exp(theta_i) / (np.exp(theta_i)+np.exp(beta_j))
    theta = theta - theta_t * lr
    beta = beta - beta_t * lr


    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    theta = np.zeros(len(data["user_id"]))
    beta = np.zeros(len(data["question_id"]))

    val_acc_lst = []
    train_ll = []
    val_ll = []

    for i in range(iterations):
        neg_lld_train = neg_log_likelihood(data, theta=theta, beta=beta)
        neg_lld_val = neg_log_likelihood(val_data, theta=theta, beta=beta)
        train_ll.append(neg_lld_train)
        val_ll.append(neg_lld_val)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        theta, beta = update_theta_beta(data, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst, train_ll, val_ll


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    lr = 0.01
    iteration = 100
    h1 = irt(train_data, val_data, lr, iteration)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
    #####################################################################
    theta = h1[0]
    beta = h1[1]
    #b
    train_ll = h1[3]
    val_ll = h1[4]
    plt.plot(list(range(iteration)), train_ll, label="loglikelihood for training set")
    plt.plot(list(range(iteration)), val_ll, label="loglikelihood for validation set")
    plt.xlabel("iteration")
    plt.ylabel("log likelihood")
    plt.legend()
    plt.show()
    #c
    test_acc = evaluate(test_data, theta, beta)
    val_acc = evaluate(val_data, theta, beta)
    print(f"learning rate is {lr}, iteration is {iteration}, accuracy for validation set is {val_acc}, accuracy for test set is {test_acc} ")
    #d
    theta = np.sort(theta)
    for i in range(3):
        beta_j = beta[i]
        cij = sigmoid(theta - beta_j)
        plt.plot(theta, cij, label=f"question {i}")
    plt.title("probability of the correct response with theta")
    plt.xlabel("theta")
    plt.ylabel("probability")
    plt.legend()
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
