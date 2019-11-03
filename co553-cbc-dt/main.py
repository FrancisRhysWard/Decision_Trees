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
    # collect_measures(dataset)
