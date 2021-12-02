import random
import neural_network_improved as nn
from starter_code.utils import *
from torch.autograd import Variable
import torch


def chose_sample(data,train_matrix, k, seed=random.randint(0, 100)):
    random.seed(seed)
    num_user, num_question = train_matrix.shape
    new_data = np.empty((num_user,num_question))
    new_data[:] = np.NaN
    for i in np.random.choice(k, int(np.floor(k))):
        new_data[data["user_id"][i]][data["question_id"][i]] = data["is_correct"][i]
    return new_data




def predict(model,train_data, valid_data):
    model.eval()
    pred = []
    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        pred.append(1 if guess == valid_data["is_correct"][i] else 0)
    return pred


# def evaluate(data, learn_rate, iteration, validation_data):
#     """
#     This function will evaluate the give data, and print out the accuracy
#     :param data: The data that we are trying to evaluate
#     :param learn_rate: The learning rate that we will be using on the item response
#     :param iteration: The iterations we will be using
#     :param validation_data: The validation data that we will use to evaluate the data
#     :return: none
#     """



# def find_lr(data, validation_data, learn_rate, iteration):
#     """
#     :param data: The data we want to find the best learning rate
#     :param validation_data: The validation data that we use to implement item response
#     :param learn_rate: the list of learning rate that will be implemented in ir
#     :param iteration: the iterations we use for ir, this will stay the same for all ir, so that we stay constant to get
#     the best learning rate
#     :return the learning rate in the list of learning rates that generate the best outcome
#     """
#     results = np.zeros(shape=len(learn_rate))
#     count = 0
#     for i in learn_rate:
#         theta, beta, _, _, _ = ir.irt(data, validation_data, i, iteration)
#         results[count] = ir.evaluate(validation_data, theta, beta)
#         count += 1
#     return learn_rate[np.argmax(results)]

# def find_num_points(data):
#     count = 0
#     for i in range(data.shape[0]):
#         for j in range(data.shape[1]):
#             if data[i][j] ==1 or data[i][j]==0:
#                 count+=1
#     return count

if __name__ == "__main__":
    sparse_matrix = load_train_sparse("../data").toarray()
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")
    # sparse_matrix = torch.FloatTensor(sparse_matrix)
    k = len(train_data["user_id"])
    data1 = chose_sample(train_data, sparse_matrix, k, 30)
    data2 = chose_sample(train_data, sparse_matrix, k, 60)
    data3 = chose_sample(train_data, sparse_matrix, k, 90)
    samples = [data1, data2, data3]
    num_question = sparse_matrix.shape[1]
    best_lr = 0.01
    best_num_epoch = 50
    k1 = 100
    k2 = 10
    best_lamb = 0.001

    model = nn.AutoEncoder(num_question, k1, k2)


    result = []
    for i in samples:
        train_matrix = i.copy()
        zero_train_matrix = i.copy()
        zero_train_matrix[np.isnan(train_matrix)] = 0
        zero_train_matrix = torch.FloatTensor(zero_train_matrix)
        train_matrix = torch.FloatTensor(train_matrix)
        train_list1, val_list1 = nn.train(model, best_lr, train_matrix, zero_train_matrix, val_data, best_num_epoch, lamb=best_lamb)
        result.append(predict(model, zero_train_matrix, val_data))
    print(result)
    pred = []
    for i in np.arange(len(result[0])):
        temp = (result[0][i] + result[1][i] + result[2][i]) / 3
        pred.append(temp >= 0.5)
    acc = np.sum((val_data["is_correct"] == np.array(pred))) \
          / len(val_data["is_correct"])
    print(acc)

