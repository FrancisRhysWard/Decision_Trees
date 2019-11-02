from evaluate import evaluate
from assign_unique_id import assign_id


def prune(tree, pruned_tree, validation_data):
    '''

    :param tree: a built tree that needs a haircut
    :param validation_data: 200x8 numpy array
    :return: a tree with a fresh haircut
    '''

    assign_id(tree.node_list)
    assign_id(pruned_tree.node_list)

    current_tree = tree

    stopPruning = False

    # Evaluate the current tree
    # current_accuracy = evaluate(validation_data, current_tree)[0]
    while not stopPruning:

        reassignCurrentTree = False

        node_list = pruned_tree.node_list

        current_accuracy = evaluate(validation_data, current_tree)[0]
        # print(' ====================================== STARTING OVER ======================================')

        # Starting with the last layer
        for layer_idx, layer in enumerate(node_list[::-1]):

            # layer_idx indicates the index of the current layer in consideration
            layer_idx = len(node_list) - layer_idx - 1

            # print('---------------  LAYER {}  -----------------'.format(layer_idx))
            if layer_idx == 0:
                stopPruning = True

            # Look at each node
            for node_idx, node in enumerate(layer[::-1]):

                node_idx = len(layer) - node_idx - 1

                # If a node has no children - cannot prune - continue
                if node.children is None:
                    continue

                # If a node has children - attempt pruning - evaluate
                elif node.children[0].children is None and node.children[1].children is None:

                    # print('Before: \n {} \n'.format(pruned_tree.node_list))

                    # Make back up of the current pruned tree
                    children_under_consideration = node.children.copy()
                    previous_pruned_node_list = pruned_tree.node_list.copy()

                    # Remove children from the node_list (+1 because children are in next layer)
                    pruned_node_id = node.id
                    node_list[layer_idx + 1].remove(node.children[0])
                    node_list[layer_idx + 1].remove(node.children[1])

                    # Set children to None
                    node.children = None

                    # print('After: \n {} \n'.format(pruned_tree.node_list))

                    # Evaluate based on the validation data
                    pruned_accuracy = evaluate(validation_data, pruned_tree)[0]

                    # Has accuracy increased?
                    if pruned_accuracy > current_accuracy:

                       # print('BEFORE {}  ----- >  AFTER {}'.format(current_accuracy, pruned_accuracy))

                        # Also Remove children from the current tree node_list
                        for current_tree_node in current_tree.node_list[layer_idx]:

                            if current_tree_node.id == pruned_node_id:

                                if current_tree_node.children[0].children is None and current_tree_node.children[1].children is None:

                                    print('Pruning from layer {}, node {}'.format(layer_idx, node_idx))
                                    # Delete the same node's children in the current tree
                                    current_tree.node_list[layer_idx + 1].remove(current_tree_node.children[0])
                                    current_tree.node_list[layer_idx + 1].remove(current_tree_node.children[1])

                                    # Set children to None
                                    current_tree_node.children = None

                                    reassignCurrentTree = True

                                    break

                        if reassignCurrentTree is True:
                            break

                    else:
                        # Revert back the changes
                        for child in children_under_consideration:
                            pruned_tree.node_list[layer_idx + 1].append(child)
                        node.children = children_under_consideration
                        # print('Reverting: \n {} \n'.format(pruned_tree.node_list))
                        # print('Reverting children: {}'.format(node.children))

                        continue


                    # If a node has no grandchildren - delete children


                else:
                    continue


            if reassignCurrentTree is True:
                break


    return current_tree
