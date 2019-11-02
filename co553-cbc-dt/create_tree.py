from tree import DecisionTree


def create_tree(_dataset, _max_depth):
    '''

    :param _dataset: dataset that will be assigned to the root of the tree
    :param _max_depth: REMOVE THIS
    :return: return a DecisionTree object (not trained)
    '''
    return DecisionTree(_dataset, _max_depth)



def decision_tree_learning(tree):
    '''
    Perform the learning process
    :param tree: tree object
    :return: trained tree object
    '''

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
        for child in tree.node_list[-1]:
            if len(child.dataset) == 0:
                continue

            child.find_split()
            child.split_data()
            child.create_children()

        if old_node_list == tree.node_list:
            new_nodes_being_added = False

    return tree
