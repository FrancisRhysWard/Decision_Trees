from evaluate import cross_validation
from prune_validate import prune_validation


def collect_measures(dataset):
    '''

    :param dataset:
           measures = [
                        classification rate,
                        confusion matrix,
                        max_depth,
                        {label, precision, recall, f1}
                        ]
    :return:
    '''

    # Collect performance measures of unpruned trees (x10 trees)
    unpruned_measures10 = cross_validation(dataset)

    # Collect performance measures of pruned trees (x90 trees)
    pruned_measures90 = prune_validation(dataset)



    unpruned_classificationRate = [measure[0] for measure in unpruned_measures10]





