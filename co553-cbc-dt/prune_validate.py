import numpy as np
from tree import DecisionTree
from evaluate import *
from create_tree import create_tree
from pruning import prune

clean_dataset = np.loadtxt("./wifi_db/clean_dataset.txt")

##noisy_data = np.loadtxt("./co553-cbc-dt/wifi_db/noisy_dataset.txt")


def prune_validation(data):
      '''
      prunes and evaluates
      '''

      divided_data = divide_data(data, 10) # shuffles then divides data

      avg_errors = []

      for i in range(10):
        test_data = divided_data[i]  ##  loop over test data sets

        errors_on_this_test = []

        for j in range(1,10):  ## loop over validation and training
            validation_data = divided_data[(i+j) % 10]

            training_data = np.concatenate([ a for a in divided_data if not (a==test_data).all() and not (a==validation_data).all()])


            print(f' training data  {training_data}, type {type(training_data)}, length                  {len(training_data)}, shape {training_data.shape}')
            tree = create_tree(training_data, 10)

            ##prune tree on validation data

            pruned_tree = prune(tree, validation_data)

            errors_on_this_test.append(1 - evaluate(test_data, pruned_tree)[0])

        avg_err_on_this_test = sum(errors_on_this_test) / 9
        avg_errors.append(avg_err_on_this_test)
      total_error = sum(avg_errors) / 10

      return total_error

            ##

if __name__ == "__main__":
    print(prune_validation(clean_dataset))


