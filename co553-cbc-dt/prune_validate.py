import numpy as np
from tree import DecisionTree
from evaluate import *
from create_tree import create_tree, decision_tree_learning
from pruning3 import prune

clean_dataset = np.loadtxt("./wifi_db/clean_dataset.txt")
noisy_dataset = np.loadtxt("./wifi_db/noisy_dataset.txt")



def print_results(tree):

    for layer in tree.node_list:
        print('\n\n')
        sum = 0
        for node in layer:
            print("LAYER #{} ---> Node #{} has an attribute: {} with dataset of length: {}".format(tree.node_list.index(layer), layer.index(node), node.split_attribute[1][2:],node.dataset.shape))

            # if tree.node_list.index(layer) < 5:
            #     with open('layer-{}-node{}.csv'.format(tree.node_list.index(layer), layer.index(node)), 'w+') as file:
            #         for line in node.dataset:
            #
            #             file.write(str(line) + '\n')

            # try:
            #     print("\t ---- Its children are {}".format(len(node.children)))
            # except:
            #     pass
            # # If attribute is None, show the dataset
            # if None in node.split_attribute[1][2:]:
            #     print(node.dataset)

            if len(set([sample[-1] for sample in node.dataset])) == 1:
                print('This node has one single label =======================================  {}'.format(node.dataset[0][-1]))
            #
            if node.children == None:
                print('This node has no children!')
                print(node.dataset)

            sum += node.dataset.shape[0]
        print("\t ---- Total shape summation: 2000 = {}".format(sum))



def prune_validation(data):

    # Shuffle and divide data
    divided_data = divide_data(data, 10) # shuffles then divides data
    avg_errors = []

    unpruned_measures = []
    all_90_measures = []

    for i in range(10):
        print('Splitting on testing')
        # Split the test data
        test_data = divided_data[i]  ##  loop over test data sets
        errors_on_this_test = []

        for j in range(1,10):  ## loop over validation and training
            print('Splitting validation')
            # Split the data
            validation_data = divided_data[(i+j) % 10]
            training_data = np.concatenate([ a for a in divided_data if not (a==test_data).all() and not (a==validation_data).all()])

            # Train a tree
            tree = create_tree(training_data, 10)
            decision_tree_learning(tree)
            tree_copy = create_tree(training_data, 10)
            decision_tree_learning(tree_copy)
            # print('Unpruned tree has {} layers.'.format(len(tree.node_list)))

            # Error on unpruned tree
            # unpruned_measures.append(evaluate(test_data, tree)[0])

            # Prune tree on validation data
            pruned_tree = prune(tree, tree_copy, validation_data)

            # print(pruned_tree.node_list)
            # counter = 0
            # for layer in pruned_tree.node_list:
            #     if layer != []:
            #         counter+=1
            # print('Pruned tree has {} layers.'.format(counter))
            #print(evaluate(clean_dataset, pruned_tree))
            errors_on_this_test.append(1 - evaluate(test_data, pruned_tree)[0])

            measures = evaluate(test_data, pruned_tree)

            all_90_measures.append(measures)

        unpruned_measures.append(evaluate(test_data, tree)[0])

        # Collect the statistics
        avg_err_on_this_test = sum(errors_on_this_test) / len(errors_on_this_test)
        avg_errors.append(avg_err_on_this_test)

        total_error = sum(avg_errors) / len(avg_errors)

    return all_90_measures, unpruned_measures


if __name__ == "__main__":
    pruned_results = prune_validation(clean_dataset)

    av_acc, av_cm, av_depth = get_avg_stats(pruned_results)

    print(av_acc, av_depth, av_cm)

    for room in room_labels:
        p = precision_recall(room, av_cm)[0]
        r = precision_recall(room, av_cm)[1]

        #print(f"Room Lable = {room} & precision = {round(p,4)} & recall = {round(r,4)} & F1 = {round(F1(p, r),4)} \\\\")

        print(f"{room} & {round(p,4)} &  {round(r,4)} &  {round(F1(p, r),4)} \\\\")

