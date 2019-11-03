import numpy as np
import matplotlib.pyplot as plt
from tree import DecisionTree
from random import choice, randint
from predict import predict
from evaluate import evaluate, divide_data
from pruning3 import prune
from prune_validate import prune_validation
from create_tree import create_tree, decision_tree_learning
from print_tree import print_results


# Import the data
clean_dataset = np.loadtxt("./wifi_db/clean_dataset.txt")
noisy_dataset = np.loadtxt("./wifi_db/noisy_dataset.txt")



if __name__ == '__main__':

    # Choose the dataset
    current_dataset = noisy_dataset
    divided_data = divide_data(current_dataset, 10)
    i = 0
    j = 1

    test_data = divided_data[i]
    errors_on_this_test = []

    validation_data = divided_data[(i + j) % 10]
    training_data = np.concatenate([a for a in divided_data if not (a == test_data).all() and not (a == validation_data).all()])



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