import numpy as np
from random import choice, randint
counter = 0
import scipy
import sklearn
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, precision_recall_curve

import graphviz



# dataset = np.loadtxt('./wifi_db/clean_dataset.txt')
# # noisy_dataset = np.loadtxt('./wifi_db/noisy_dataset.txt')
#
# Y = dataset[:, -1]
# X = dataset[:, 0:-1]
# print(X)

# X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size=0.33, random_state=42)
#
# clf = tree.DecisionTreeClassifier()
# clf = clf.fit(X_train, y_train)
#
#
# y_dt = clf.predict(X_test)
#
# print(accuracy_score(y_test, y_dt))
#
# tree.export_graphviz(clf)


clean_dataset = np.loadtxt("./wifi_db/clean_dataset.txt")

data_size = len(clean_dataset)

wifi_attr = [i for i in range(1, len(clean_dataset[0]))]

print(clean_dataset[:10])



def p_k(k, S):
    '''
    takes a label k and array of data S and returns the sample probability of k in S as a float
    '''

    S_size = len(S)
    number_k_in_S = len([s for s in S if s[-1] == k]) ## number of samples in S with label k

    # print(number_k_in_S, S_size)
    return number_k_in_S / S_size
## print(p_k(1.0, clean_dataset[:10]))

def H(S):
    '''
    Entropy of a dataset S
    returns float
    '''

    labels = [1,2,3,4]
    # print(len(S))
    list_of_p = [p_k(k, S) for k in labels if p_k(k, S) != 0] ## create ps so they only need to be calcd once
    summands = [p * np.log2(p) for p in list_of_p]  ## create H summands
    return - sum(summands)

## print(H(clean_dataset))

def split(S):
    print("Taking data of shape {} ---> {}".format(S.shape, counter+1))

    random_attribute_index = choice(range(len(S[0]) - 1)) ## chooses a random wifi signal
   ## rand_data_subset = clean_dataset[randint(0, 1000): randint(1000, -1)]

    size = len(S)

    if len(set([sample[-1] for sample in S])) == 1:
        print("NO SPLIT")
        return (np.array([]), np.array([]), None, None)

    if len(S) == 0:
        print("NO SPLIT")
        return (np.array([]), np.array([]), None, None)

    rand_data_subset = S[int(size * 0.2): int(size * 0.8)]

    values = [sample[random_attribute_index] for sample in rand_data_subset]

    rand_split_value = choice(values)

    S_left = np.array([sample for sample in S if sample[random_attribute_index] <= rand_split_value])

    S_right = np.array([sample for sample in S if sample[random_attribute_index] > rand_split_value])

    return S_left, S_right, wifi_attr[random_attribute_index], rand_split_value


def remainder(S_l_r):
    S_left, S_right = S_l_r[:2]
    if len(S_right) == 0:
        return 0
    size_left = len(S_left[0])
    size_right = len(S_right)
    # print("Sizes are: left {}, right {}".format(size_left, size_right))
    return (size_left / (size_left + size_right) * H(S_left)) + (size_right / (size_left + size_right) * H(S_right))


def Gain(S, split):
    return H(S) - remainder(split)




class DecisionTree():

    def __init__(self, dataset, max_depth):

        self.max_depth = max_depth
        self.start_node = Node(self, dataset, parent=None)

        # Initialise the database of nodes
        self.node_list = list()
        self.node_list.append([self.start_node])


    def change_node_list(self, children, depth):
        '''
        Adds children to the correct layer (layer number = depth)

        :param children:
        '''

        # Try to append children to the specified depth
        try:
            for child in children:
                self.node_list[depth].append(child)
        # If it does not yet exist, create a new layer
        except:
            self.node_list.append(children)


    def update_max_depth(self, new_depth):
        if new_depth > self.max_depth:
            self.max_depth = new_depth


    def learn(self):
        pass



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

        # Attributes set to None as default
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

        # if self.dataset.shape == (0,0):
        #     self.split_attribute = None

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

        # print("DATASET 1 is of type: {}".format(type(dataset1)))
        # print("DATASET 2 is of type: {}".format(type(dataset2)))


        # If no further splits can be made
        if len(dataset1) == 0 and len(dataset2) == 0:
            self.children = None
            return

        child1 = Node(self.tree, dataset1, self)
        child2 = Node(self.tree, dataset2, self)

        self.children = [child1, child2]

        # Add them to the tree.node_list
        self.tree.change_node_list(self.children, self.depth+1)

        # Update the max_depth of the tree
        self.tree.update_max_depth(self.depth+1)


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

    print(type(start_node.dataset))
    print(type(start_node.children[0].dataset))
    print(start_node.dataset.shape)
    print(start_node.children[0].dataset.shape)

    # Repeat for each node
    for layer in tree.node_list: # while new layers are added
        print('*'*40)
        print('Current layer contains: {}'.format(len(layer)))
        print('*'*40)

        # Run through each child in a layer
        for child in layer:
            child.find_split()
            child.split_data()
            child.create_children()


    # child = start_node.children[0]
    # print(child.dataset.shape)
    # print(type(child.dataset))
    # print(child.dataset)
    # child.find_split()
    # child.split_data()
    # child.create_children()



if __name__ == '__main__':

    max_depth = 10

    # Create a tree
    tree = create_tree(clean_dataset, max_depth)

    run_learning(tree)