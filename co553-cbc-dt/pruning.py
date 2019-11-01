from evaluate import evaluate


def prune(tree, pruned_tree, validation_data):
    '''

    :param tree: a built tree that needs a haircut
    :param validation_data: 200x8 numpy array
    :return: a tree with a fresh haircut
    '''

    current_tree = tree

    # Evaluate the current tree
    # current_accuracy = evaluate(validation_data, current_tree)[0]

    node_list = pruned_tree.node_list

    stopPruning = False

    while not stopPruning:

        current_accuracy = evaluate(validation_data, current_tree)[0]

        # Starting with the last layer
        for layer_idx, layer in enumerate(node_list[::-1]):

            # layer_idx indicates the index of the current layer in consideration
            layer_idx = len(node_list) - layer_idx - 1

            if layer_idx == 0:
                stopPruning = True

            # Look at each node
            for node_idx, node in enumerate(layer):

                # print('Considering layer#{} and node#{}'.format(layer_idx, node_idx))


                # If a node has no children - cannot prune - continue
                if node.children is None:
                    # print('                                 -----> no children :(')
                    continue

                # If a node has children - attempt pruning - evaluate
                else:

                    # If a node has no grandchildren - delete children
                    if node.children[0].children is None and node.children[1].children is None:
                        # Remove children from the node_list
                        node_list[layer_idx + 1].remove(node.children[0])
                        node_list[layer_idx + 1].remove(node.children[1])

                        # Set children to None
                        node.children = None

                        # Evaluate based on the validation data
                        pruned_accuracy = evaluate(validation_data, pruned_tree)[0]

                        # Has accuracy increased?
                        if pruned_accuracy > current_accuracy:

                            # Also Remove children from the current tree node_list
                            current_tree.node_list[layer_idx + 1].remove(current_tree.node_list[layer_idx][node_idx].children[0])
                            current_tree.node_list[layer_idx + 1].remove(current_tree.node_list[layer_idx][node_idx].children[1])

                            # Set children to None
                            current_tree.node_list[layer_idx][node_idx].children = None



                        else:
                            continue

                    else:
                        continue

                if stopPruning:
                    break

            if stopPruning:
                break

        # stopPruning = True


    return current_tree