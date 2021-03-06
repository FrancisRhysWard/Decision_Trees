from random import choice, randint
import numpy as np
import find_split



clean_dataset = np.loadtxt("./wifi_db/clean_dataset.txt")
noisy_dataset = np.loadtxt("./wifi_db/noisy_dataset.txt")

data_size = len(clean_dataset)
wifi_attr = [i for i in range(1, len(clean_dataset[0]))]



class Node():
    '''
    Class Node (leaf is a node with no children)
    '''

    def __init__(self, tree, dataset, parent, children=list(), split_attribute=None):
        '''

        :param tree: tree object to which the object belongs        (object DecisionTree)
        :param dataset: dataset that is contained within the node   (numpy array)
        :param parent: parent node                                  (object Node)
        :param children: list of children nodes                     (list of Node objects)
        :param split_attribute: tuple (Wifi attribute, Split value) (tuple)
        '''

        self.tree = tree

        # Initialise dataset
        self.dataset = dataset

        # The predominant label in the dataset
        self.label = None

        # Initalise node parent
        self.parent = parent

        # Depth is initialised automatically given the parent node
        # If it's the first node
        if self.parent == None:
            self.depth = 0
        else:
            self.depth = self.parent.depth + 1

        # Assign children
        self.children = children

        self.coord = [0,0]

        # Tuple --> (feature, split value)
        self.split_attribute = split_attribute


    def find_split(self):
        '''
        Finds the best split_attribute for the particular self.dataset

        :param: self.dataset
        :return: self.split_attribute --> Tuple (feature, split value)
                                      --> if None, we do not proceed
        '''

        self.split_attribute = find_split.find_split(self.dataset)

        return self.split_attribute


    def split_data(self):
        '''
        Dataset is split according to the provided split criterion

        :param self.dataset, self.split_attribute (if == (None, None), then do not split further)
        :return: _split_dataset1, _split_dataset2
        '''

        _split_dataset_left, _split_dataset_right = self.split_attribute[1][:2]

        return _split_dataset_left, _split_dataset_right


    def create_children(self):
        '''
        After the split_attribute is determined, we split the data accordingly and initialise children

        :param: self.dataset
        :return: None
        '''

        dataset1, dataset2 = self.split_data() # if (None, None) -->

        # If no further splits can be made
        if dataset1 is None or dataset2 is None:
            self.children = None
            return



        child1 = Node(self.tree, dataset1, self)
        child2 = Node(self.tree, dataset2, self)

        self.children = [child1, child2]

        # Add them to the tree.node_list
        self.tree.change_node_list(self.children, self.depth+1)
