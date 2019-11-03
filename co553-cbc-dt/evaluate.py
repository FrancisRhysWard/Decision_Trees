import numpy as np
from random import choice, randint
from predict import predict
from create_tree import create_tree, decision_tree_learning



room_labels = [1,2,3,4]


def divide_data(data, k_folds):
    '''
    Shuffles and splits data into k-folds
    :param data: dataset (np.array)
    :param k_folds: number of splits
    :return: list of np.arrays
    '''

    np.random.shuffle(data)

    return np.split(data, k_folds)


def precision_recall(room, confusion_matrix):
    '''
    Takes room label and confusion matrix and returns the precision and recall

    :param room: label
    :param confusion_matrix: confusion matrix
    :return: precision, recall
    '''

    tp = confusion_matrix[room - 1][room - 1]
    fp = confusion_matrix.sum(0)[room - 1] - tp
    fn = confusion_matrix.sum(1)[room - 1] - tp

    # if tp + fp == 0:
    #     print(tp, fp)

    precision =  tp / (tp + fp)
    recall = tp / (tp + fn)

    return precision, recall


def F1(precision, recall):
    '''
    Takes the precision and recall and returns f1 measure
    :param precision: precision
    :param recall: recall
    :return: f1 measure
    '''

    return 2 * (precision * recall) / (precision + recall)


def create_confusion_matrix(test_data, tree_predictions):
    '''
    Creates a confusion matrix based on TEST and predicted labels
    :param test_data: TEST
    :param tree_predictions: TEST with replaced labels by predictions
    :return: confusion matrix (np.array)
    '''

    # Initialise confusion matrix
    cm_dimensions = (4, 4)

    # Create a numpy representation
    confusion_matrix = np.zeros(cm_dimensions)

    # Add all results to confusion matrix
    for sample_actual, sample_predicted in zip(test_data, tree_predictions):
        confusion_matrix[int(sample_actual[-1]) - 1][int(sample_predicted[-1]) - 1] += 1

    return confusion_matrix



def evaluate(test_data, learned_tree):
    '''
    Takes a test array and a learned tree and returns test measures
    :param test_data: data on which the tree is tested
    :param learned_tree: tree object
    :return: measures --> [
                            classification rate,
                            confusion matrix,
                            max_depth,
                            {label, precision, recall, f1}
                            ]
    '''

    # Initialise some variables
    total_size = len(test_data)
    tree_predictions = []
    test_data_to_predict = test_data.copy()


    # Retrieve TEST with labels replaced by predictions
    for sample in test_data_to_predict:
        tree_predictions.append(predict(learned_tree, sample))  # should = test_data with actual labels replace with predictions

    tree_predictions = np.array(tree_predictions)


    # Create confusion matrix
    confusion_matrix = create_confusion_matrix(test_data, tree_predictions)


    # Calculate statistics
    classification_rate = np.trace(confusion_matrix) / total_size
    max_depth = len(learned_tree.node_list)


    # Initialise a list of measures with classification rate and max_depth
    measures = [classification_rate, confusion_matrix, max_depth]

    # Add all other stats to measures
    for i in room_labels:
        room_measures = {}
        room_measures["label"] = i
        room_measures["precision"] = precision_recall(i, confusion_matrix)[0]
        room_measures["recall"] = precision_recall(i, confusion_matrix)[1]
        room_measures["f1"] = F1(precision_recall(i, confusion_matrix)[0], precision_recall(i, confusion_matrix)[1])
        measures.append(room_measures)


    return measures



def cross_validation(data):
    '''
    Takes some data and performs cross validation to iterate over test and training data and train different trees to return the average performance
    :param data:
    :return:
    '''

    divided_data = divide_data(data, 10)

    #print(divided_data)

    all_10_measures = []

    for i in range(10):

        test_data = divided_data[i]

        training_data = np.concatenate([ a for a in divided_data if not (a==test_data).all()])

        tree = create_tree(training_data, 2)

        learned_tree = decision_tree_learning(tree)

        measures = evaluate(test_data, learned_tree)

        all_10_measures.append(measures)

        print(measures[2])

    return all_10_measures



def get_avg_stats(cv):

    avg_acc = sum([measure[0] for measure in cv])/len(cv)

    avg_cm = sum([measure[1] for measure in cv])/len(cv)

    avg_depth = sum([measure[2] for measure in cv])/len(cv)


    return avg_acc, avg_cm, avg_depth


if __name__ == "__main__":

    noisy_dataset = np.loadtxt("./wifi_db/noisy_dataset.txt")

    clean_dataset = np.loadtxt("./wifi_db/clean_dataset.txt")
    #print(cross_validation(clean_dataset))

    av_acc, av_cm, av_depth = get_avg_stats(cross_validation(clean_dataset))

    print(av_acc, av_depth, av_cm)

    for room in room_labels:
        p = precision_recall(room, av_cm)[0]
        r = precision_recall(room, av_cm)[1]
        #print(f"Room Lable = {room} & precision = {round(p,4)} & recall = {round(r,4)} & F1 = {round(F1(p, r),4)} \\\\")
        print(f"{room} & {round(p,4)} &  {round(r,4)} &  {round(F1(p, r),4)} \\\\")

