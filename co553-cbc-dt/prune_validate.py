import numpy as np
from tree import DecisionTree
from evaluate import *
from create_tree import create_tree, run_learning
from pruning import prune

clean_dataset = np.loadtxt("./wifi_db/clean_dataset.txt")
noisy_dataset = np.loadtxt("./wifi_db/noisy_dataset.txt")



def print_results(tree):

    for layer in tree.node_list:
        print('\n\n')
        sum = 0
        for node in layer:
            print("LAYER #{} ---> Node #{} has an attribute: {} with dataset of length: {}".format(tree.node_list.index(layer), layer.index(node), node.split_attribute[1][2:],node.dataset.shape))

            # if tree.node_list.index(layer) < 5:
            #     with open('layer-{}-node{}.csv'.format(tree.node_list.index(layer), layer.index(node)), 'w+') as file:
            #         for line in node.dataset:
            #
            #             file.write(str(line) + '\n')

            # try:
            #     print("\t ---- Its children are {}".format(len(node.children)))
            # except:
            #     pass
            # # If attribute is None, show the dataset
            # if None in node.split_attribute[1][2:]:
            #     print(node.dataset)

            if len(set([sample[-1] for sample in node.dataset])) == 1:
                print('This node has one single label =======================================  {}'.format(node.dataset[0][-1]))
            #
            if node.children == None:
                print('This node has no children!')
                print(node.dataset)

            sum += node.dataset.shape[0]
        print("\t ---- Total shape summation: 2000 = {}".format(sum))



def prune_validation(data):

    # Shuffle and divide data
    divided_data = divide_data(data, 10) # shuffles then divides data
    avg_errors = []

    for i in range(10):

        # Split the test data
        test_data = divided_data[i]  ##  loop over test data sets
        errors_on_this_test = []

        for j in range(1,10):  ## loop over validation and training

            # Split the data
            validation_data = divided_data[(i+j) % 10]
            training_data = np.concatenate([ a for a in divided_data if not (a==test_data).all() and not (a==validation_data).all()])

            # Train a tree
            tree = create_tree(training_data, 10)
            run_learning(tree)
            tree_copy = create_tree(training_data, 10)
            run_learning(tree_copy)

            # Prune tree on validation data
            pruned_tree = prune(tree, tree_copy, validation_data)
            print(evaluate(clean_dataset, pruned_tree))
            errors_on_this_test.append(1 - evaluate(test_data, pruned_tree)[0])

        # Collect the statistics
        avg_err_on_this_test = sum(errors_on_this_test) / len(errors_on_this_test)
        avg_errors.append(avg_err_on_this_test)

        total_error = sum(avg_errors) / len(avg_errors)

    return total_error


if __name__ == "__main__":
    print(prune_validation(clean_dataset))

