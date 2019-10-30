

def predict(tree, sample):

    node = tree.start_node

    # print('Evaluating sample: {}'.format(sample))

    while node.children is not None:

        attribute, value = node.split_attribute[1][2:]

        # print('Current node has attribute {} with value {}'.format(attribute, value))

        # If eg. X[0] < 30
        if sample[attribute-1] <= value:
            # print('<----<----<----<--')
            node = node.children[0]  # left child

        # If eg. X[0] >= 30
        else:
            # print('>---->---->---->--')
            node = node.children[1]  # right child


    # Print the label once you are at a leaf
    # print('Final node dataset: {}'.format(node.dataset))
    # print('Classified as: {}'.format(node.dataset[0][-1]))

    # Assign the predicted label
    old_label = sample[-1]
    sample[-1] = node.dataset[0][-1]
    new_label = sample[-1]

    # if new_label != old_label:
    #     print('INCORRECTLY CLASSIFIED')

    if sample[-1] == node.dataset[0][-1]:
        return sample

