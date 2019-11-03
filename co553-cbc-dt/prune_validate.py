import numpy as np
from tree import DecisionTree
from evaluate import *
from create_tree import create_tree, decision_tree_learning
from pruning import prune
from print_tree import print_results

clean_dataset = np.loadtxt("./wifi_db/clean_dataset.txt")
noisy_dataset = np.loadtxt("./wifi_db/noisy_dataset.txt")


def prune_validation(data):
    '''
    1. Split data into TEST and TRAINING+VALIDATION (x10 times)
        2. Split TRAINING+VALIDATION into TRAINING and VALIDATION (x9 times)
            3. For each TRAINING and VALIDATION:
                a) Train a tree using TRAINING
                b) Prune a tree using VALIDATION
                c) Test each pruned tree using TEST (9 trees x 10 test datasets = 90 measures)

    :param data: full dataset (clean_dataset OR noisy_dataset)
    :return: all_90_measures: list of [measures1, measures2, ..., measures90]
                                    where measure1 = [classification_rate, ...]
    '''

    # Shuffle and divide data
    divided_data = divide_data(data, 10)

    avg_errors = []
    all_90_measures = []

    for i in range(10):
        # Split TEST and TRAINING+VALIDATION (x10 times)
        test_data = divided_data[i]
        errors_on_this_test = []

        # Split TRAINING+VALIDATION ---> TRAINING and VALIDATION (x9 times)
        for j in range(1,10):

            validation_data = divided_data[(i+j) % 10]
            training_data = np.concatenate([ a for a in divided_data if not (a==test_data).all() and not (a==validation_data).all()])

            print(i,j)

            # Train a tree
            tree = create_tree(training_data)
            decision_tree_learning(tree)

            # Prune tree on VALIDATION
            pruned_tree = prune(tree, validation_data)

            # Calculate error of pruned tree on TEST
            errors_on_this_test.append(1 - evaluate(test_data, pruned_tree)[0])

            # Evaluate pruned on TEST
            measures = evaluate(test_data, pruned_tree)

            # Collect all measures of the pruned tree
            all_90_measures.append(measures)

        # Collect error stats
        avg_err_on_this_test = sum(errors_on_this_test) / len(errors_on_this_test)
        avg_errors.append(avg_err_on_this_test)
        total_error = sum(avg_errors) / len(avg_errors)


    return all_90_measures


if __name__ == "__main__":

    # Sandbox
    pruned_results = prune_validation(noisy_dataset)

<<<<<<< HEAD
    av_acc, av_cm, av_depth, _, _ = get_avg_stats(pruned_results)
=======
    av_acc, av_cm, av_depth, min_depth, max_depth = get_avg_stats(pruned_results)
>>>>>>> 0dee2c2ad29f441c4973f5d7ef94fb03a22b9108

    print(av_acc, av_depth, av_cm, min_depth, max_depth)

    for room in room_labels:
        p = precision_recall(room, av_cm)[0]
        r = precision_recall(room, av_cm)[1]

        #print(f"Room Lable = {room} & precision = {round(p,4)} & recall = {round(r,4)} & F1 = {round(F1(p, r),4)} \\\\")

        print(f"{room} & {round(p,4)} &  {round(r,4)} &  {round(F1(p, r),4)} \\\\")

