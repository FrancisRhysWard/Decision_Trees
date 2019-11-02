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

    stopPruning = False

    node_list = pruned_tree.node_list

    # Evaluate the current tree
    # current_accuracy = evaluate(validation_data, current_tree)[0]
    while not stopPruning:

        print(' ====================================== STARTING OVER ======================================')

        current_accuracy = evaluate(validation_data, pruned_tree)[0]

        # Starting with the last layer
        for layer_idx, layer in enumerate(node_list[::-1]):

            # layer_idx indicates the index of the current layer in consideration
            layer_idx = len(node_list) - layer_idx - 1

            print('---------------  LAYER {}  -----------------'.format(layer_idx))

            if layer_idx == 0:
                stopPruning = True

            # Look at each node
            for node_idx, node in enumerate(layer[::-1]):

                node_idx = len(layer) - node_idx - 1

                print('Node_idx = {}'.format(node_idx))

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

                        print('BEFORE {}  ----- >  AFTER {}'.format(current_accuracy, pruned_accuracy))

                        current_accuracy = evaluate(validation_data, pruned_tree)[0]
                        
                    else:
                        # Revert back the changes
                        for child in children_under_consideration:
                            pruned_tree.node_list[layer_idx + 1].append(child)
                        node.children = children_under_consideration
                        node_list = pruned_tree.node_list
                        # print('Reverting: \n {} \n'.format(pruned_tree.node_list))
                        print('Reverting children: {} and {}'.format(node.children[0].id, node.children[1].id))

                        continue


                    # If a node has no grandchildren - delete children


                else:
                    continue


    return pruned_tree
