import numpy as np
import matplotlib.pyplot as plt
from tree import DecisionTree
from random import choice, randint
from predict import predict
from evaluate import evaluate, divide_data
<<<<<<< HEAD
from pruning import prune
=======
<<<<<<< HEAD
=======
from pruning3 import prune
>>>>>>> bb5ed4c18b6dd4d25e9eeeb3c4018308a85c139f
>>>>>>> 0dee2c2ad29f441c4973f5d7ef94fb03a22b9108
from prune_validate import prune_validation
from create_tree import create_tree, decision_tree_learning
from print_tree import print_results
from collect_measures import collect_measures


# Import the data
clean_dataset = np.loadtxt("./wifi_db/clean_dataset.txt")
noisy_dataset = np.loadtxt("./wifi_db/noisy_dataset.txt")

data_names = ['clean', 'noisy']


if __name__ == '__main__':

    for name_idx, dataset in enumerate([clean_dataset, noisy_dataset]):
        collect_measures(dataset, data_names[name_idx])


    # ======= YOUR DATA HERE =======

<<<<<<< HEAD
    # dataset = np.load(" INPUT PATH TO YOUR DATASET ")
    # collect_measures(dataset)
=======


    tree = create_tree(training_data, 10)
    decision_tree_learning(tree)

    print(len(tree.node_list))
    print(tree.node_list)

    prune(tree, validation_data )

    print(len(tree.node_list))
    print(tree.node_list)


    # # Retrieve all measures
    # all_measures = prune_validation(current_dataset)
    #
    # # Get the average classification rate
    # sum_accuracy = 0
    # for measure in all_measures:
    #     accuracy = measure[0]
    #     sum_accuracy += accuracy
    #
    # print('Total observed accuracy on NOISY is {}%'.format((sum_accuracy/90)*100))




    # measures, unpruned_measures = prune_validation(clean_dataset)
    # rates = [rate[0] for rate in measures]
    # print(rates)
    # test_rates = []
    # unpruned_rates = []
    # temp1 = []
    # temp2 = []
    #
    # for j in range(1, 91):
    #     temp1.append(rates[j-1])
    #     temp2.append(rates[j - 1])
    #     if j % 9 == 0:
    #         test_rates.append(temp1)
    #         unpruned_rates.append(temp2)
    #         temp1 = []
    #         temp2 = []
    #
    #
    # axs_num = 10
    # fig, axs = plt.subplots(1, axs_num, sharey=True, tight_layout=True)
    #
    # # We can set the number of bins with the `bins` kwarg
    # for i in range(axs_num):
    #     axs[i].hist(np.array(test_rates[i]))
    #     axs[i].set_title('{0:.2f}'.format(np.mean(np.array(unpruned_rates[axs_num]))))
    #
    # plt.show()
    #
    #
    # # for splits_on_test in range(10):
    # #
    # #     for splits_on_validation in range(9):
    # #
    # #         tempRates.append(measures90)
>>>>>>> 0dee2c2ad29f441c4973f5d7ef94fb03a22b9108
