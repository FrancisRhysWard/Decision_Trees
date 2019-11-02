import numpy as np
import matplotlib.pyplot as plt
from tree import DecisionTree
from random import choice, randint
from predict import predict
from evaluate import evaluate, divide_data
from pruning import prune
from prune_validate import prune_validation
from create_tree import create_tree, decision_tree_learning


# Import the data
clean_dataset = np.loadtxt("./wifi_db/clean_dataset.txt")
noisy_dataset = np.loadtxt("./wifi_db/noisy_dataset.txt")


def print_results(tree):
    '''
    Prints out useful information about the structure of a trained tree
    '''

    print(tree.node_list)
    i = 0
    for layer in tree.node_list:
        if layer != []:
            i=+1

    print(i)
    # for layer in tree.node_list:
    # #     print('\n\n')
    # #     sum = 0
    # #
    #     print('NUMBER OF LAYERS: {}'.format(len(tree.node_list)))
    # #
    #     for node in layer:
    # #
    # #
    #         # print("LAYER #{} ---> Node #{} has an attribute: {}".format(tree.node_list.index(layer), layer.index(node), node.split_attribute[1][2:],node.dataset.shape))
    # #
    # #         try:
    # #             print("\t ---- Its children are {}".format(len(node.children)))
    # #         except:
    # #             pass
    # #         # If attribute is None, show the dataset
    # #         if None in node.split_attribute[1][2:]:
    # #             print(node.dataset)
    #         if node.children is None:
    #             if len(set([sample[-1] for sample in node.dataset])) == 1:
    #                 print('This node has one single label =======================================  {}'.format(node.dataset[0][-1]))
    #             else:
    #                 print('ERROR'*30)
    #                 print(node.dataset)

            # if node.children == None:
            #     print('This node has no children!')
            #     print(node.dataset)

        #     sum += node.dataset.shape[0]
        # print("\t ---- Total shape summation: 2000 = {}".format(sum))


if __name__ == '__main__':




    print(prune_validation(noisy_dataset))

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