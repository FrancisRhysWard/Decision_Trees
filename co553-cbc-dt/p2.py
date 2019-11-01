import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines

from tree import DecisionTree

from main2 import *



clean_dataset = np.loadtxt("./wifi_db/clean_dataset.txt")

tree = create_tree(clean_dataset, 10)

run_learning(tree)


start_node = tree.start_node

print(tree.node_list)

node = start_node


for layer in tree.node_list:
    child_count = 0
    for node in layer:
        node_x = layer.index(node)
        node_y =  - tree.node_list.index(layer)
        if node.children != None:
            for child in node.children:
                xt = [node_x,  child_count]
                yt = [node_y, node_y - 1]
                plt.plot(xt, yt)
                child_count += 1

plt.show()
