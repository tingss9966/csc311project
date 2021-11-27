import math
import random

import numpy as np

import item_response as ir
import matplotlib.pyplot as plt
from starter_code.utils import *


def chose_sample(data, k, seed):
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
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = ir.sigmoid(x)
        pred.append(1 if p_a >= 0.5 else 0)
    return pred


def find_lr(data, val_data, lr, iterations):
    results = np.zeros(shape=len(lr))
    count=0
    for i in lr:
        theta, beta, _,_,_ = ir.irt(data, val_data, i, iterations)
        results[count] = ir.evaluate(val_data, theta, beta)
        count+=1
    return lr[np.argmax(results)]

if __name__ == "__main__":
    sparse_matrix = load_train_sparse("../data")
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")
    k = len(train_data["user_id"])
    data1 = chose_sample(train_data, k, 30)
    data2 = chose_sample(train_data, k, 60)
    data3 = chose_sample(train_data, k, 90)
    samples = [data1, data2, data3]

    # Finding the best learning rate
    # You can comment out this part to save time and use the learning rate already found
    # for item response
    lrs = [0.05,0.03,0.01,0.005,0.001]
    iterations = 20
    lr = find_lr(train_data,val_data,lrs, iterations)
    # lr = 0.01
    # iterations = 20
    result = []
    for i in samples:
        theta, beta, _, _, _ = ir.irt(i, val_data, lr, iterations)
        result.append(predict(val_data, theta, beta))
    pred = []
    for i in np.arange(len(result[0])):
        temp = (result[0][i] + result[1][i] + result[2][i]) / 3
        pred.append(temp >= 0.5)
    acc = np.sum((val_data["is_correct"] == np.array(pred))) \
          / len(val_data["is_correct"])
    print(acc)


