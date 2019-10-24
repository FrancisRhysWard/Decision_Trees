import numpy as np
import node
from tree import DecisionTree
from random import choice, randint


clean_dataset = np.loadtxt("./wifi_db/clean_dataset.txt")

data_size = len(clean_dataset)

wifi_attr = [i for i in range(1, len(clean_dataset[0]))]


def create_tree(_dataset, _max_depth):
    return DecisionTree(_dataset, _max_depth)


def run_learning(tree):

    # Take starting node
    start_node = tree.start_node

    # Find the perfect split
    start_node.find_split()

    # Split the data
    start_node.split_data()

    # Create children
    start_node.create_children()

    # Run a loop while new nodes are being added
    new_nodes_being_added = True
    while new_nodes_being_added:

        old_node_list = tree.node_list.copy()
        # Repeat for each node (excluding the starting node)
        # Starting with last nodes
        # Run through each child in a layer

        print('*' * 40)
        print('Current layer #{} contains: {} elements'.format(len(tree.node_list), len(tree.node_list[-1])))
        print('*' * 40)

        for child in tree.node_list[-1]:
            if len(child.dataset) == 0:
                continue

            child.find_split()
            child.split_data()
            child.create_children()

        if old_node_list == tree.node_list:
            new_nodes_being_added = False


    print_results()


def print_results():

    for layer in tree.node_list:
        print('\n\n')
        sum = 0
        for node in layer:
            print("LAYER #{} ---> Node #{} has an attribute: {} with dataset of length: {}"
                  .format(tree.node_list.index(layer), layer.index(node), node.split_attribute[1][2:], node.dataset.shape))
            # try:
            #     print("\t ---- Its children are {}".format(len(node.children)))
            # except:
            #     pass
            # # If attribute is None, show the dataset
            # if None in node.split_attribute[1][2:]:
            #     print(node.dataset)

            sum += node.dataset.shape[0]
        print("\t ---- Total shape summation: 2000 = {}".format(sum))


if __name__ == '__main__':

    max_depth = 10

    # Create a tree
    tree = create_tree(clean_dataset, max_depth)

    run_learning(tree)