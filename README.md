# Introduction to Machine Learning - Coursework 1 (Decision Trees)

Authors: Rhys, James, Richard, Alex

Date: 3 November 2019


## Requirements

This project uses only numpy and matplotlib libraries.


## Installing

1. Clone the repository to your Desktop.
2. Open `main.py`
3. Import your data by specifying the path to your dataset

    ```
        # ======= YOUR DATA HERE =======
    
        # dataset = np.load(" INPUT PATH TO YOUR DATASET ")
    ```
4. Run `main.py` to collect all the measures for your dataset. 

Please refer to **Project Structure** section to learn more about what `main.py` does.

 

## Tree and node representation

We thought the best way to represent trees and their respective constituents is to employ object-oriented representation.

In our project, a tree is an object of class `DecisionTree`. Its nodes are objects of class `Node`.



### Class DecisionTree

**Attributes:**

* `self.start_node` : root of the tree
* `self.node_list` : [[node0], [node11, node12], [...], ...]; each list represents a layer of a tree containing
all node objects in that layer.


### Class Node

**Attributes:**

* `self.tree`: tree object to which the node belongs
* `self.dataset`: dataset corresponding to the node
* `self.label`: predominant label in node's dataset
* `self.parent`: parent node
* `self.children`: children nodes [node1, node2]
* `self.coord`: node coordinate (used to plot the tree)
* `self.split_attribute`: all the information about splitting (such as wifi attribute or attribute value)


**Methods:**

* `def find_split(self)`: finds and returns split attributes
* `def split_data(self)`: splits node's data according to split attribute and returns its children's datasets
* `def create_children(self)`: creates children with respective datasets




## Project Structure

The project folder contains all the methods required for:
1. Creating an instance of a DecisionTree
2. Training a DecisionTree on training data
3. Performing the splitting of the datasets (into test, training, validation sets)
4. Performing cross validation 
5. Pruning a tree based on some validation set
6. Collect all the performance measures (classification rate, confusion matrix, precision, etc.)
7. Visualising a decision tree


#### main.py:

This is the root file of our project. In its default form, it runs the function `collect_measures(dataset)`,
which prints out all the calculated measures as output, as well as useful figures on the distribution of
measurements in `./Figures`.


#### collect_measures.py:

Prints all the statistical information about the performance of both *pruned* and *unpruned* trees.

For example, here is a sample output on *clean data*:

```
======================== UNPRUNED (Clean Dataset) =========================


1. Average classification accuracy: 0.974%
2. Average max depth: 13.6 layers
3. Average confusion matrix: 

     [[ 49.6   0.    0.4   0. ]
     
      [  0.   48.1   1.9   0. ]
     
      [  0.2   1.9  47.7   0.2]
     
      [  0.5   0.    0.1  49.4]]

4. Label-specific stats:
>> Room 1
	Precision: 0.9860834990059641
	Recall: 0.992
	F1: 0.989

>> Room 2
	Precision: 0.9620000000000001
	Recall: 0.9620000000000001
	F1: 0.962

>> Room 3
	Precision: 0.9520958083832336
	Recall: 0.954
	F1: 0.953

>> Room 4
	Precision: 0.9959677419354838
	Recall: 0.988
	F1: 0.992

===========================================================================

```


`collect_measures.py` indirectly employs all of the functionality of our project. Let's now review 
it step-by-step.

It also includes function `def plot_histogram(data, n_plots, file_name, normalising=False)`, which produces the afore-mentioned figures.

#### create_tree.py:

Contains functions to **instantiate** a tree object, as well as run **learning**.

* `def create_tree(_dataset)`: returns an object DecisionTree
* `def decision_tree_learning(tree)`: runs learning based on the methods of `Node` class, such as `find_split()`,
 `split_data()`, `create_children()`.



#### evaluate.py:

Contains the following functions:

* `def evaluate(test_data, learned_tree)`: takes a test dataset and evaluates a learned tree against it

* `def cross_validation(data)`: takes a dataset and performs a 10-fold cross-validation returning performance measures

* `def divide_data(data, k_folds)`: shuffles and divides data into k-folds

* `def create_confusion_matrix(test_data, tree_predictions)`: creates a confusion matrix based on the true and predicted labels
of a dataset

* `def precision_recall(room, confusion_matrix)`: calculates precision and recall measures given a label and confusion matrix

* `def F1(precision, recall)`: calculates f1 measure given precision and recall measures

* `def get_avg_stats(all_10_measures)`: returns averages of classification accuracy, confusion matrix and
tree depth, as well as minimum depth, maximum depth and label-specific measures (precision, recall, f1).
avg_cm, avg_depth, min_depth, max_depth, room_stats



#### predict.py:

Contains `def predict(tree, sample)`, which, given a learned tree and data sample, outputs
the predicted label (based on the split attributes of the learned tree).


#### prune_validate.py:

Contains:

* `def prune_validation(data)`: performs comprehensive evaluation of pruned decision trees (implicitly implements
pruning). Here's a step-by-step functionality:
    * Split data into **TEST** and **TRAINING+VALIDATION** datasets (x10 times)
        * Split **TRAINING+VALIDATION** into *TRAINING* and *VALIDATION* (x9 times)
        * For each, *TRAINING* and *VALIDATION*:
            1. Train a tree using *TRAINING*
            2. Prune a tree using *VALIDATION*
            3. Test each pruned tree using **TEST** 
    * Return 90 measures
      - (9 pruned trees x 10 test sets = 90 performance measures)
        
        
 We must now discuss the pruning implementation.
 
 
 
#### pruning.py:

This file contains function `prune(pruned_tree, validation_data)`, which takes in an unpruned tree and validation set as inputs
and attempts to prune the tree. Here are the steps of pruning:

1. Iterate through every layer starting from the deepest
2. Iterate through all the nodes in a layer
3. Under the condition that a node has children but no grandchildren:
    4. Attempt pruning the leaves of the node (children)
    5. Compare the accuracy of the tree before and after removing the leaves
        6. If accuracy increased or didn't change, keep the pruning
        7. If accuracy decreased, revert back the pruning
8. Delete empty layers (empty lists)
9. Return pruned tree        


#### plot_tree.py:

Plots a decision tree before and after pruning
































    

