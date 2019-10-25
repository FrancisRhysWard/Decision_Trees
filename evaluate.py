import numpy as np
from random import choice, randint

clean_dataset = np.loadtxt("./co553-cbc-dt/wifi_db/clean_dataset.txt")

##noisy_data = np.loadtxt("./co553-cbc-dt/wifi_db/noisy_dataset.txt")

wifi_labels = [1,2,3,4]


def divide_data(data):
    '''
    takes a data set (np.array), shuffles it then splits it into ten arrays and returns a list of these arrays
    '''
    np.random.shuffle(data) ## shuffles data

    return np.split(data, 10)  ## returns list of 10 np arrays used as CV folds


def precision_recall(wifi, cm):
    '''
    takes a wifi label and confusion matrix and returns the precision and recall
    '''
    tp = cm[wifi - 1][wifi - 1]
    fp = cm.sum(0)[wifi - 1] - tp
    fn = cm.sum(1)[wifi - 1] -  tp

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

    tree_predictions = ## should = test_data with actual labels replace with predictions

    cm_dimensions = (4,4)

    confusion_matrix = np.zeros(cm_dimensions)

    for sample in range(total_size):
        confusion_matrix[test_data[sample[-1] - 1]][tree_predictions[sample[-1] - 1] += 1 ## adds one to corect col/row of cm

    classification_rate = np.trace(confusion_matrix) / total_size

    measures = [classification_rate]

    for i in wifi_labels:
        wifi_measures = {}
        wifi_measures["label"] = i
        wifi_measures["precision"] = precision_recall(i, cm)[0]
        wifi_measures["recall"] = precision_recall(i, cm)[1]
        wifi_measures["f1"] = F1(precision_recall(i, cm)[0], precision_recall(i, cm)[1])

        measures.append(wifi_measures)


    ## number_incorrect_predictions = set.difference(set(test_data), set(tree_predictions))


    return measures

def cross_validation(data):
    '''
    takes some data and performs corss validation to iterate over test and training data and train different trees to return the average performance
    '''

    divided_data = divide_data(data)

    for i in range(10):
        test_data = divided_data[i]

        ##validation_data = divided_data[(i+1) % 10]

        ##training_data = np.array(divided_data[(i+2) % 10: i]).flatten()

        training_data = np.array(divided_data[i+1 % 10 : i]).flatten()



if __name__ == "__main__":
    cross_validation(clean_dataset)


