from collections import Counter

def predict(tree, sample):

    node = tree.start_node

    # print('Evaluating sample: {}'.format(sample))

    while node.children is not None:

        attribute, value = node.split_attribute[1][2:]

        # If eg. X[0] < 30
        if sample[attribute-1] <= value:
            node = node.children[0]  # left child

        # If eg. X[0] >= 30
        else:
            node = node.children[1]  # right child


    # Assign the predicted label
    node_dataset_labels = [sample[-1] for sample in node.dataset]
    _set_of_labels = set(node_dataset_labels)
    # print(node_dataset_labels)
    label_occurences = []

    for element in _set_of_labels:
        label_occurences.append((element, node_dataset_labels.count(element)))

    # print(label_occurences)

    predominant_label = max(label_occurences, key=lambda x: x[1])[0]
    sample[-1] = predominant_label

    return sample

