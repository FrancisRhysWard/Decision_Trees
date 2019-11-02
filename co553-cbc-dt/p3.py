import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines

from tree import DecisionTree

from main2 import *



clean_dataset = np.loadtxt("./wifi_db/clean_dataset.txt")

tree = create_tree(clean_dataset, 10)

run_learning(tree)


start_node = tree.start_node

width = 2000

#print(tree.node_list)

max_nodes_in_layer = max([len(layer) for layer in tree.node_list])

for layer in tree.node_list:
    for i,node in enumerate(layer):
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

