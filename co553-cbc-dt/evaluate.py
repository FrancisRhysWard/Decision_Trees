import numpy as np
from random import choice, randint
from predict import predict



clean_dataset = np.loadtxt("./wifi_db/clean_dataset.txt")

##noisy_data = np.loadtxt("./co553-cbc-dt/wifi_db/noisy_dataset.txt")

wifi_labels = [1,2,3,4]


def divide_data(data, n):
    '''
    takes a data set (np.array), shuffles it then splits it into ten arrays and returns a list of these arrays
    '''
    np.random.shuffle(data) ## shuffles data

    return np.split(data, n)  ## returns list of n np arrays used as CV folds


def precision_recall(wifi, cm):
    '''
    takes a wifi label and confusion matrix and returns the precision and recall
    '''
    tp = cm[wifi - 1][wifi - 1]
    fp = cm.sum(0)[wifi - 1] - tp
    fn = cm.sum(1)[wifi - 1] -  tp

    # if tp + fp == 0:
    #     print(tp, fp)

    precision =  tp / (tp + fp)
    recall = tp / (tp + fn)

    return precision, recall


def F1(p, r):
    '''
    takes the precision and recall and returns f1 measure
    '''

    return 2 * ( p * r ) / ( p + r )


def evaluate(test_data, learned_tree):
    '''
    takes a test array and a learned tree and returns test measures
    '''

    total_size = len(test_data)

    tree_predictions = []

    test_data_to_predict = test_data.copy()


    for sample in test_data_to_predict:
        tree_predictions.append(predict(learned_tree, sample))  # should = test_data with actual labels replace with predictions

    tree_predictions = np.array(tree_predictions)

    cm_dimensions = (4,4)

    confusion_matrix = np.zeros(cm_dimensions)

    # Add all results to confusion matrix
    for sample_actual, sample_predicted in zip(test_data, tree_predictions):

        confusion_matrix[int(sample_actual[-1])-1][int(sample_predicted[-1]) - 1] += 1

    classification_rate = np.trace(confusion_matrix) / total_size

    # print('Classification rate: {}%'.format(classification_rate * 100))
    measures = [classification_rate]

    for i in wifi_labels:
        wifi_measures = {}
        wifi_measures["label"] = i
        wifi_measures["precision"] = precision_recall(i, confusion_matrix)[0]
        wifi_measures["recall"] = precision_recall(i, confusion_matrix)[1]
        wifi_measures["f1"] = F1(precision_recall(i, confusion_matrix)[0], precision_recall(i, confusion_matrix)[1])

        measures.append(wifi_measures)

    # print(confusion_matrix)
    ## number_incorrect_predictions = set.difference(set(test_data), set(tree_predictions))


    return measures

def cross_validation(data):
    '''
    takes some data and performs corss validation to iterate over test and training data and train different trees to return the average performance
    '''

    divided_data = divide_data(data, 10)

    all_10_measures = []

    for i in range(10):
        test_data = divided_data[i]

        ##validation_data = divided_data[(i+1) % 10]

        ##training_data = np.array(divided_data[(i+2) % 10: i]).flatten()

        training_data = np.array(divided_data[i+1 % 10 : i]).flatten()

        tree = create_tree(training_data)

        learned_tree = run_learning(tree)

        measures = evaluate(test_data, learned_tree)

        all_10_measures.append(measures)

    print(all_10_measures)

if __name__ == "__main__":
    cross_validation(clean_dataset)


