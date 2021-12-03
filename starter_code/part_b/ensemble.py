import random

import matplotlib.pyplot as plt

import neural_network_improved as nn
from starter_code.utils import *
from torch.autograd import Variable
import torch


def chose_sample(data, train_matrix, k, seed=random.randint(0, 100)):
    """
    This helper function is for chosing samples for bagging
    :param data: The training data in dictionary form
    :param train_matrix: The training data in 2D sparse matrix
    :param k: The number of train that each sample uses
    :param seed: the random seed for the chose
    :return the new chosen data
    """
    random.seed(seed)
    num_user, num = train_matrix.shape
    new_data = np.empty((num_user, num))
    new_data[:] = np.NaN
    for i in np.random.choice(k, int(np.floor(k))):
        new_data[data["user_id"][i]][data["question_id"][i]] = data["is_correct"][i]
    return new_data


def predict(model, train_data, valid_data):
    """
    This function will predict the outcome of the data given
    :param model: the model we train and test on
    :param train_data: The training data that we use to predict
    :param valid_data: The validation data that we test on
    :return the prediction list
    """
    model.eval()
    pred = []
    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        pred.append(1 if guess == valid_data["is_correct"][i] else 0)
    return pred


def get_acc(result, num, valid_data):
    """
    Get the accuracy of the result given
    :param result: a 2D array that contains the lists of lists of results
    :param num: the number of lists in the sample
    :param valid_data: the data to check on
    :return the accuracy
    """
    pred = []
    for i in np.arange(len(result[0])):
        temp = 0
        for j in range(num):
            temp += result[j][i]
        temp = temp / num
        pred.append(temp >= 0.5)
    acc = np.sum((valid_data["is_correct"] == np.array(pred))) \
          / len(valid_data["is_correct"])
    return acc


def evaluate(the_model, sample, lr, valid_data, tests_data, epoch, lamb):
    """
    Evaluate and find the accuracy
    :param the_model: The neural network model to be trained on
    :param sample: The list of sample data
    :param lr: the learning rate
    :param valid_data: the validation data that we will check our accuracy
    :param tests_data: the test dat that we check our accuracy on
    :param epoch: the number of epochs
    :param lamb: the best lambda
    :return the validation accuracy and the test accuracy
    """
    result = []
    test_result = []
    for i in sample:
        train_matrix = i.copy()
        zero_train_matrix = i.copy()
        zero_train_matrix[np.isnan(train_matrix)] = 0
        zero_train_matrix = torch.FloatTensor(zero_train_matrix)
        train_matrix = torch.FloatTensor(train_matrix)
        nn.train(the_model, lr, train_matrix, zero_train_matrix, valid_data, epoch,
                 lamb=lamb)
        result.append(predict(the_model, zero_train_matrix, valid_data))
        test_result.append(predict(the_model, zero_train_matrix, tests_data))
    acc = get_acc(result, len(sample), valid_data)
    test_acc = get_acc(test_result, len(sample), tests_data)
    return acc, test_acc


def find_lr(the_model, data, validation_data, learn_rate, num_epoch, lamb,tests_data):
    """
    To find the best learning rate
    """
    results = []
    for i in learn_rate:
        val, test = evaluate(the_model,data,i,validation_data,tests_data,num_epoch,lamb)
        results.append(val)
    return results


def find_num_points(data):
    """
    debugging helper function, to find out how many non repeating training example in the data
    """
    count = 0
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i][j] == 1 or data[i][j] == 0:
                count += 1
    return count


if __name__ == "__main__":
    sparse_matrix = load_train_sparse("../data").toarray()
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")




    k = len(train_data["user_id"])
    data1 = chose_sample(train_data, sparse_matrix, k, 30)
    data2 = chose_sample(train_data, sparse_matrix, k, 60)
    data3 = chose_sample(train_data, sparse_matrix, k, 90)
    samples = [data1, data2, data3]
    num_question = sparse_matrix.shape[1]
    best_lr = 0.01
    best_num_epoch = 30
    k1 = 100
    k2 = 10
    best_lamb = 0.001
    model = nn.AutoEncoder(num_question, k1, k2)

    # lr = [0.1,0.05,0.03,0.01,0.005,0.003,0.001]
    # result = find_lr(model,samples,val_data,lr,best_num_epoch,best_lamb,test_data)
    # plt.plot(lr, result, label="ensemble")
    # results = []
    # for i in lr:
    #     zero_train_matrix = sparse_matrix.copy()
    #     zero_train_matrix[np.isnan(sparse_matrix)] = 0
    #     zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    #     temp = torch.FloatTensor(sparse_matrix)
    #     nn.train(model, best_lr, temp, zero_train_matrix, val_data, best_num_epoch,
    #                                    lamb=best_lamb)
    #     acc = nn.evaluate(model,zero_train_matrix,val_data)
    #     results.append(acc)
    # plt.plot(lr, results, label="nn")
    # plt.legend()
    # plt.show()
    # plt.show()

    accuracy, test_accuracy = evaluate(model, samples, best_lr, val_data, test_data, best_num_epoch, best_lamb)
    print(f"Validation accuracy is {accuracy}")
    print(f"Test accuracy is {test_accuracy}")
