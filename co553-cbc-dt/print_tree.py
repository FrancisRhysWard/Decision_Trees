
def print_results(tree):
    '''
    Prints out useful information about the structure of a trained tree
    '''

    print(tree.node_list)
    i = 0
    for layer in tree.node_list:
        if layer != []:
            i=+1

    print(i)
    # for layer in tree.node_list:
    # #     print('\n\n')
    # #     sum = 0
    # #
    #     print('NUMBER OF LAYERS: {}'.format(len(tree.node_list)))
    # #
    #     for node in layer:
    # #
    # #
    #         # print("LAYER #{} ---> Node #{} has an attribute: {}".format(tree.node_list.index(layer), layer.index(node), node.split_attribute[1][2:],node.dataset.shape))
    # #
    # #         try:
    # #             print("\t ---- Its children are {}".format(len(node.children)))
    # #         except:
    # #             pass
    # #         # If attribute is None, show the dataset
    # #         if None in node.split_attribute[1][2:]:
    # #             print(node.dataset)
    #         if node.children is None:
    #             if len(set([sample[-1] for sample in node.dataset])) == 1:
    #                 print('This node has one single label =======================================  {}'.format(node.dataset[0][-1]))
    #             else:
    #                 print('ERROR'*30)
    #                 print(node.dataset)

            # if node.children == None:
            #     print('This node has no children!')
            #     print(node.dataset)

        #     sum += node.dataset.shape[0]
        # print("\t ---- Total shape summation: 2000 = {}".format(sum))