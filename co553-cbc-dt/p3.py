import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines
from evaluate import *


from tree import DecisionTree

from main2 import *



clean_dataset = np.loadtxt("./wifi_db/clean_dataset.txt")


# Shuffle and divide data
divided_data = divide_data(clean_dataset, 10) # shuffles then divides data
i = 0
j = 1

# Split the test data
test_data = divided_data[i]  ##  loop over test data sets

# Split the data
validation_data = divided_data[(i+j) % 10]
training_data = np.concatenate([ a for a in divided_data if not (a==test_data).all() and not (a==validation_data).all()])

# Train a tree
tree = create_tree(training_data, 10)
decision_tree_learning(tree)
tree_copy = create_tree(training_data, 10)
decision_tree_learning(tree_copy)

start_node = tree.start_node

width = 2000


# WORKING SOLUTION

max_nodes_in_layer = max([len(layer) for layer in tree.node_list])

for layer in tree.node_list:
    for i, node in enumerate(layer):
        if node.children != None:
            node.children[0].coord[0] = node.coord[0]  - width  #len(tree.node_list)/(tree.node_list.index(layer) +1)
            node.children[0].coord[1] = node.coord[1] - 20 # -1 depth
            node.children[1].coord[0] = node.coord[0]  + width  #len(tree.node_list)/(tree.node_list.index(layer)+1)
            node.children[1].coord[1] = node.coord[1] - 20 # -1 depth
    width = width * 0.5

for layer in tree.node_list:
    for node in layer:
        node_x = node.coord[0]
        node_y = node.coord[1]
        if node.children != None:
            for child in node.children:
                xt = [node_x,  child.coord[0]]
                yt = [node_y, child.coord[1]]
                plt.plot(xt, yt)

plt.show()

