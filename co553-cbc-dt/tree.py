from node import Node
from random import choice, randint
import numpy as np



class DecisionTree():

    def __init__(self, dataset, max_depth):
        '''

        :param dataset: root dataset of the tree
        :param max_depth: REMOVE THIS
        '''

        self.max_depth = max_depth

        # Create the root node
        self.start_node = Node(self, dataset, parent=None)

        # Initialise the database of nodes
        self.node_list = list()
        self.node_list.append([self.start_node])


    def change_node_list(self, children, depth):
        '''
        Adds children to the correct layer

        :param children: list of two Node objects [Child1, Child2]
        '''

        # Try to append children to the specified depth
        try:
            for child in children:
                self.node_list[depth].append(child)
        # If it does not yet exist, create a new layer
        except:
            self.node_list.append(children.copy())


    def update_max_depth(self, new_depth):
        '''
        REMOVE THIS?
        :param new_depth:
        :return:
        '''
        if new_depth > self.max_depth:
            self.max_depth = new_depth
