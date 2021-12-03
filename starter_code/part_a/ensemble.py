import random
import item_response as ir
from starter_code.utils import *
import matplotlib.pyplot as plt

def chose_sample(data, k, seed=random.randint(0, 100)):
    """
    This function will chose the data that will be used for training, this is called bagging
    :param data: The whole data set that we will be choosing from
    :param k: This is the ratio of the data that we will be using from the original
    :param seed: This is the random seed, in case we don't want similar data being generated
    :return The data chosen after using bagging
    """
    data = data.copy()
    sample = {
        "user_id": [],
        "question_id": [],
        "is_correct": []
    }
    random.seed(seed)
    for i in np.random.choice(k, int(np.floor(k))):
        sample["user_id"].append(data["user_id"][i])
        sample["question_id"].append(data["question_id"][i])
        sample["is_correct"].append(data["is_correct"][i])
    return sample


def predict(data, theta, beta):
    """
    We will predict the data with the given theta and beta, this is using item response.
    :param data: the data that we are trying to predict
    :param theta: the theta that we use to predict
    :param beta: the beta we use to predict
    :return a list that shows which data that we have predicted correctly
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = ir.sigmoid(x)
        pred.append(1 if p_a >= 0.5 else 0)
    return pred


def get_acc(val_result, validation_data):
    """
        Get the accuracy of the result given
        :param val_result: a 2D array that contains the lists of lists of results
        :param validation_data: the data to check on
        :return the accuracy
        """
    pred = []
    for i in np.arange(len(val_result[0])):
        temp1 = (val_result[0][i] + val_result[1][i] + val_result[2][i]) / 3
        pred.append(temp1 >= 0.5)
    acc = np.sum((validation_data["is_correct"] == np.array(pred))) \
          / len(validation_data["is_correct"])
    return acc


def evaluate(data, learn_rate, iteration, validation_data,test_data):
    """
    This function will evaluate the give data, and print out the accuracy
    :param data: The data that we are trying to evaluate
    :param learn_rate: The learning rate that we will be using on the item response
    :param iteration: The iterations we will be using
    :param validation_data: The validation data that we will use to evaluate the data
    :param test_data: the test data that we will use to evaluate the data
    :return: none
    """
    val_result = []
    test_result = []
    for i in data:
        theta, beta, _, _, _ = ir.irt(i, validation_data, learn_rate, iteration)
        val_result.append(predict(validation_data, theta, beta))
        test_result.append(predict(test_data, theta, beta))
    val_acc = get_acc(val_result, validation_data)
    test_acc = get_acc(test_result, test_data)
    return val_acc, test_acc


def find_lr(data, validation_data, learn_rate, iteration, test_data):
    """
    :param data: The data we want to find the best learning rate
    :param validation_data: The validation data that we use to implement item response
    :param learn_rate: the list of learning rate that will be implemented in ir
    :param iteration: the iterations we use for ir, this will stay the same for all ir, so that we stay constant to get
    the best learning rate
    :return the learning rate in the list of learning rates that generate the best outcome
    """
    results = []
    for i in learn_rate:
        val,test = evaluate(data,i,iteration,validation_data,test_data)
        results.append(val)
    return results


if __name__ == "__main__":
    sparse_matrix = load_train_sparse("../data")
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    tests_data = load_public_test_csv("../data")
    k = len(train_data["user_id"])
    data1 = chose_sample(train_data, k, 30)
    data2 = chose_sample(train_data, k, 60)
    data3 = chose_sample(train_data, k, 90)
    samples = [data1, data2, data3]

    # To check how ensemble improves accuracy on IRT

    # lrs = [0.08, 0.05, 0.03, 0.01, 0.005, 0.001]
    # iterations = 20
    # result= find_lr(samples, val_data, lrs, iterations, tests_data)
    # plt.plot(lrs,result,label = "ensemble")
    #
    # result = []
    # for i in lrs:
    #     temp = ir.irt(train_data, val_data, i, iterations)
    #     acc = ir.evaluate(val_data, temp[0], temp[1])
    #     result.append(acc)
    # plt.plot(lrs, result,label = "irt")
    # plt.legend()
    # plt.show()
    # plt.show()

    lr = 0.01
    iterations = 20
    validation_accuracy,test_accuracy = evaluate(samples, lr, iterations, val_data, tests_data)
    print(f"Validation accuracy is {validation_accuracy}")
    print(f"Test accuracy is {test_accuracy}")
