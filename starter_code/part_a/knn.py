from matplotlib import pyplot as plt
from sklearn.impute import KNNImputer
from starter_code.utils import *


def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    nbrs = KNNImputer(n_neighbors=k)
    tran = nbrs.fit_transform(matrix.transpose())
    mat = tran.transpose()
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################
    acc_user = []
    acc_item = []
    k_list = [1, 6, 11, 16, 21, 26]
    for k in range(len(k_list)):
        acc_user.append(knn_impute_by_user(sparse_matrix, val_data, k_list[k]))
        acc_item.append(knn_impute_by_item(sparse_matrix, val_data, k_list[k]))
        print(f"user accuracy rate for k = {k_list[k]} is {acc_user[k]}")
        print(f"item accuracy rate for k = {k_list[k]} is {acc_item[k]}")
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    user_max = 0
    item_max = 0
    user_max_i = 0
    item_max_i = 0
    for i in range(len(k_list)):
        if acc_user[i] > user_max:
            user_max = acc_user[i]
            user_max_i = i
        if acc_item[i] > item_max:
            item_max = acc_item[i]
            item_max_i = i
    acc_user_test = knn_impute_by_user(sparse_matrix, test_data, k_list[user_max_i])
    acc_item_test = knn_impute_by_item(sparse_matrix, test_data, k_list[item_max_i])
    print(f" best user accuracy rate is {user_max} when k = {k_list[user_max_i]} on validation set, using the same k \n"
          f"on test set we will get accuracy of {acc_user_test}")
    print(f" best item accuracy rate is {item_max} when k = {k_list[item_max_i]} on validation set, using the same k \n"
          f"on test set we will get accuracy of {acc_item_test}")
    plt.plot(k_list, acc_user, label="user accuracy with k")
    plt.plot(k_list, acc_item, label="item accuracy with k")
    plt.xlabel("k")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
