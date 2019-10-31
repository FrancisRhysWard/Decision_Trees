from tree import DecisionTree


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

        # print('*' * 40)
        # print('Current layer #{} contains: {} elements'.format(len(tree.node_list), len(tree.node_list[-1])))
        # print('*' * 40)

        for child in tree.node_list[-1]:
            if len(child.dataset) == 0:
                continue

            child.find_split()
            child.split_data()
            child.create_children()

        if old_node_list == tree.node_list:
            new_nodes_being_added = False

    return tree
    # print_results()