from random import choice, randint
import numpy as np
from main2 import wifi_attr


def p_k(k, S):
    '''
    takes a label k and array of data S and returns the sample probability of k in S as a float
    '''

    S_size = len(S)
    number_k_in_S = len([s for s in S if s[-1] == k]) ## number of samples in S with label k

    return number_k_in_S / S_size


def H(S):
    '''
    Entropy of a dataset S
    returns float
    '''

    labels = [1,2,3,4]

    list_of_p = [p_k(k, S) for k in labels if p_k(k, S) != 0] ## create ps so they only need to be calcd once
    summands = [p * np.log2(p) for p in list_of_p]  ## create H summands
    return - sum(summands)


def split(S):

    random_attribute_index = choice(range(len(S[0]) - 1)) ## chooses a random wifi signal

    size = len(S)

    if len(set([sample[-1] for sample in S])) == 1:
        return (np.array([]), np.array([]), None, None)

    if len(S) == 0:
        return (np.array([]), np.array([]), None, None)

    rand_data_subset = S[int(size * 0.2): int(size * 0.8)]

    values = [sample[random_attribute_index] for sample in rand_data_subset]

    rand_split_value = choice(values)

    S_left = np.array([sample for sample in S if sample[random_attribute_index] <= rand_split_value])

    S_right = np.array([sample for sample in S if sample[random_attribute_index] > rand_split_value])

    return S_left, S_right, wifi_attr[random_attribute_index], rand_split_value


def remainder(S_l_r):
    S_left, S_right = S_l_r[:2]
    # I ADDED THIS
    if len(S_right) == 0:
        return 0
    size_left = len(S_left[0])
    size_right = len(S_right)

    return (size_left / (size_left + size_right) * H(S_left)) + (size_right / (size_left + size_right) * H(S_right))


def Gain(S, split):
    return H(S) - remainder(split)




class Node():

    def __init__(self, tree, dataset, parent, children=list(), split_attribute=None):

        self.tree = tree

        # Initialise dataset and the parent
        self.dataset = dataset
        self.parent = parent

        # Depth is initialised automatically given the parent node
        # If it's the first node
        if self.parent == None:
            self.depth = 0
        else:
            self.depth = self.parent.depth + 1

        # Other attributes
        self.children = children

        # Tuple --> (feature, split value)
        self.split_attribute = split_attribute


    def find_split(self):
        '''
        Finds the best split_attribute for the particular self.dataset

        :param: self.dataset
        :return: self.split_attribute --> Tuple (feature, split value)
                                      --> if None, we do not proceed
        '''

        splits = [split(self.dataset) for i in range(50)]
        gains = [Gain(self.dataset, split) for split in splits]
        max_index = np.argmax(gains)
        self.split_attribute = (gains[max_index], splits[max_index])

        return self.split_attribute


    def split_data(self):
        '''
        Dataset is split according to the provided split criterion

        :param self.dataset, self.split_attribute (if == (None, None), then do not split further)
        :return: _split_dataset1, _split_dataset2
        '''

        _split_dataset_left, _split_dataset_right, _, _ = split(self.dataset)

        return _split_dataset_left, _split_dataset_right


    def create_children(self):
        '''
        After the split_attribute is determined, we split the data accordingly and initialise children

        :param: self.dataset
        :return: None
        '''

        dataset1, dataset2 = self.split_data() # if (None, None) -->

        # If no further splits can be made
        if len(dataset1) == 0 or len(dataset2) == 0:
            self.children = None
            return

        child1 = Node(self.tree, dataset1, self)
        child2 = Node(self.tree, dataset2, self)

        self.children = [child1, child2]

        # Add them to the tree.node_list
        self.tree.change_node_list(self.children, self.depth+1)

        # Update the max_depth of the tree
        self.tree.update_max_depth(self.depth+1)
