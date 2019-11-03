from evaluate import evaluate
from assign_unique_id import assign_id


def prune(pruned_tree, validation_data):
    '''

    :param pruned_tree: a built tree that needs a haircut
    :param validation_data: 200x8 numpy array
    :return: a tree with a fresh haircut
    '''

    # Assign a unique id to every node
    assign_id(pruned_tree.node_list)

    # Retrieve the node list of the tree
    node_list = pruned_tree.node_list

    stopPruning = False
    while not stopPruning:

        # Calculate accuracy before proceeding to pruning
        current_accuracy = evaluate(validation_data, pruned_tree)[0]

        # Starting with the last layer
        for layer_idx, layer in enumerate(node_list[::-1]):

            # layer_idx indicates the index of the current layer
            layer_idx = len(node_list) - layer_idx - 1

            # If at root, stop pruning
            if layer_idx == 0:
                stopPruning = True

            # If the layer is empty, delete it
            if layer == []:
                node_list.pop(layer_idx)

            # Look at each node
            for node_idx, node in enumerate(layer[::-1]):

                # node_idx = len(layer) - node_idx - 1

                # If a node has no children - cannot prune - continue
                if node.children is None:
                    continue

                # If a node has children - attempt pruning - evaluate
                elif node.children[0].children is None and node.children[1].children is None:

                    # Make back up of the current pruned tree
                    children_under_consideration = node.children.copy()

                    # Remove children from the node_list (+1 because children are in next layer)
                    node_list[layer_idx + 1].remove(node.children[0])
                    node_list[layer_idx + 1].remove(node.children[1])

                    # Set children to None
                    node.children = None


                    # Evaluate based on VALIDATION
                    pruned_accuracy = evaluate(validation_data, pruned_tree)[0]

                    # Has accuracy increased?
                    if pruned_accuracy >= current_accuracy:

                        # If yes, set the new current accuracy as reference for next prunings
                        current_accuracy = evaluate(validation_data, pruned_tree)[0]
                        
                    else:

                        # If not, revert back the changes
                        for child in children_under_consideration:
                            pruned_tree.node_list[layer_idx + 1].append(child)
                        node.children = children_under_consideration
                        node_list = pruned_tree.node_list

                        continue

                else:
                    continue


    return pruned_tree
