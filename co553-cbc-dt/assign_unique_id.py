
def assign_id(node_list):

    # Iterate through the node_list
    for layer_idx, layer in enumerate(node_list):
        for node_idx, node in enumerate(layer):

            # Assign unique id
            node.id = int(str(layer_idx) + str(node_idx))