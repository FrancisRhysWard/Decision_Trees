import numpy as np
from tree import DecisionTree
from random import choice, randint
from predict import predict
from evaluate import evaluate
from pruning import prune
from prune_validate import prune_validation
from create_tree import create_tree, run_learning


# Import the data
clean_dataset = np.loadtxt("./wifi_db/clean_dataset.txt")
noisy_dataset = np.loadtxt("./wifi_db/noisy_dataset.txt")


def print_results(tree):
    '''
    Prints out useful information about the structure of a trained tree
    '''

    for layer in tree.node_list:
        print('\n\n')
        sum = 0
        for node in layer:
            print("LAYER #{} ---> Node #{} has an attribute: {}"# with dataset of length: {}"
                .format(tree.node_list.index(layer), layer.index(node), node.split_attribute[1][2:],node.dataset.shape))

            try:
                print("\t ---- Its children are {}".format(len(node.children)))
            except:
                pass
            # If attribute is None, show the dataset
            if None in node.split_attribute[1][2:]:
                print(node.dataset)

            if len(set([sample[-1] for sample in node.dataset])) == 1:
                print('This node has one single label =======================================  {}'.format(node.dataset[0][-1]))

            if node.children == None:
                print('This node has no children!')
                print(node.dataset)

            sum += node.dataset.shape[0]
        print("\t ---- Total shape summation: 2000 = {}".format(sum))


if __name__ == '__main__':

    # tree = create_tree(clean_dataset, 10)
    #
    # learned_tree = run_learning(tree)
    #
    # print(evaluate(clean_dataset, learned_tree))


    measures = prune_validation(clean_dataset)
    rates = [rate[0] for rate in measures]
    print(rates)

    # for splits_on_test in range(10):
    #
    #     for splits_on_validation in range(9):
    # 
    #         tempRates.append(measures90)