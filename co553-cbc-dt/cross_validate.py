import numpy as np
from tree import DecisionTree
from random import choice, randint
from predict import predict
from evaluate import *
from main2 import *

clean_dataset = np.loadtxt("./wifi_db/clean_dataset.txt")

##noisy_data = np.loadtxt("./co553-cbc-dt/wifi_db/noisy_dataset.txt")



def cross_validation(data):
     '''
     takes some data and performs corss validation to iterate over test and training data and train      different trees to return the average performance
     '''

     divided_data = divide_data(data, 10)

     all_10_measures = []

     for i in range(10):
         test_data = divided_data[i]

         ##validation_data = divided_data[(i+1) % 10]

         ##training_data = np.array(divided_data[(i+2) % 10: i]).flatten()


         training_data = np.concatenate(divided_data[i+1:] + divided_data[0:i])
         # print(f' training data  {training_data}, type {type(training_data)}, length {len(training_data)}, shape {training_data.shape}')
         tree = create_tree(training_data, 10)


         learned_tree = decision_tree_learning(tree)

         measures = evaluate(test_data, learned_tree)

         all_10_measures.append(measures)

     avg_measures = [0, {'label': 1, 'precision': 0, 'recall': 0, 'f1': 0}, {'label': 2, 'precision': 0, 'recall': 0, 'f1': 0}, {'label': 3, 'precision': 0, 'recall': 0, 'f1': 0}, {'label': 4, 'precision': 0, 'recall': 0, 'f1': 0}]

     avg_measures[0] = sum([m[0] for m in all_10_measures])/len(all_10_measures)

     for d in avg_measures[1:]:
         for m in all_10_measures:
             for d2 in m[1:]:
                 if d['label'] == d2['label']:
                     d['precision'] += d2['precision']/10
                     d['recall'] += d2['recall']/10
                     d['f1'] += d2['recall']/10

     #values = [[d.values() for d in m[1:]] for m in all_10_measures]
     #print(values)

     return avg_measures


if __name__ == "__main__":
     print(cross_validation(clean_dataset))


