import numpy as np
import matplotlib.pyplot as plt
from tree import DecisionTree
from random import choice, randint
from predict import predict
from evaluate import evaluate, divide_data
from pruning import prune
from prune_validate import prune_validation
from create_tree import create_tree, decision_tree_learning
from print_tree import print_results
from collect_measures import collect_measures


# Import the data
clean_dataset = np.loadtxt("./wifi_db/clean_dataset.txt")
noisy_dataset = np.loadtxt("./wifi_db/noisy_dataset.txt")

data_names = ['clean', 'noisy']


if __name__ == '__main__':


    # COMMENT THIS OUT WHEN TESTING ON YOUR OWN DATASET
    for name_idx, dataset in enumerate([clean_dataset, noisy_dataset]):
        collect_measures(dataset, data_names[name_idx])


    # ======= YOUR DATA HERE =======
    # dataset = np.load(" INPUT PATH TO YOUR DATASET ")



    # ======= PERFORM COMPLETE EVALUATION =======
    # collect_measures(dataset)





    # ********* Side note ************

    # ======= TO CREATE A TREE =======
    # tree = create_tree(dataset)
    # decision_tree_learning(tree)


    # ======= TO EVALUATE A TREE =======
    # measures = evaluate(test_data, tree)


    # ===== OR TO USE CROSS VALIDATION 10-fold ====
    # all_10_measures = cross_validation(dataset)


    # ======= TO EVALUATE WITH PRUNING =======
    # all_90_measures = prune_validation(dataset)