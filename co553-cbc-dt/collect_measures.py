from evaluate import cross_validation
from prune_validate import prune_validation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter
import pickle
from evaluate import get_avg_stats



def plot_histogram(data, n_plots, file_name, normalising=False):
    '''

    :param data: list of data to be plotted [data1, data2, ...]
    :param n_plots: number of subplots
    :param file_name:
    :param normalising:
    :param plotting90measures:
    :return:
    '''

    fig, axs = plt.subplots(1, 2)

    # N is the count in each bin, bins is the lower-limit of the bin
    for axs_num in range(n_plots):
        N, bins, patches = axs[axs_num].hist(data[axs_num])

    for axs_num in range(n_plots):
        axs[axs_num].set_xlabel('Measure value')
        axs[axs_num].set_ylabel('Frequency')


    plt.title(' ', y=1.08)
    fig.suptitle('{}'.format(file_name), fontsize=12)
    fig.tight_layout()
    plt.savefig('./Figures/{}.png'.format(file_name))



def averageList(lst):

    return sum(lst)/len(lst)



def collect_measures(dataset, data_name=''):
    '''

    :param dataset:
           measures = [
                        classification rate,
                        confusion matrix,
                        max_depth,
                        {label1, precision, recall, f1},
                        {label2, precision, recall, f1},
                        {label3, precision, recall, f1},
                        {label4, precision, recall, f1}
                                                        ]
    :return:
    '''

    if data_name == 'noisy':
        data_name = 'Noisy Dataset'
    elif data_name == 'clean':
        data_name = 'Clean Dataset'
    else:
        data_name = 'Unknown Dataset'

    print('Working, please wait a couple of minutes...')

    # Collect performance measures of unpruned trees (x10 trees)
    unpruned_measures10 = cross_validation(dataset)

    print('Still, working...')

    # Collect performance measures of pruned trees (x90 trees)
    pruned_measures90 = prune_validation(dataset)


    # CLASSIFICATION RATES
    unpruned_classificationRate = [measure[0] for measure in unpruned_measures10]
    pruned_classificationRate = [measure[0] for measure in pruned_measures90]

    # CONFUSION MATRICES
    unpruned_confusionMatrix = [measure[1] for measure in unpruned_measures10]
    pruned_confusionMatrix = [measure[1] for measure in pruned_measures90]

    # MAX DEPTH
    unpruned_maxDepth = [measure[2] for measure in unpruned_measures10]
    pruned_maxDepth = [measure[2] for measure in pruned_measures90]


    # ######################################################################
    #                  # LABEL SPECIFIC RESULTS #
    #
    # # LABEL1
    # unpruned_label1 = [measure[3] for measure in unpruned_measures10]
    # pruned_label1 = [measure[3] for measure in pruned_measures90]
    #
    # # LABEL2
    # unpruned_label2 = [measure[4] for measure in unpruned_measures10]
    # pruned_label2 = [measure[4] for measure in pruned_measures90]
    #
    # # LABEL3
    # unpruned_label3 = [measure[4] for measure in unpruned_measures10]
    # pruned_label3 = [measure[4] for measure in pruned_measures90]
    #
    # # LABEL4
    # unpruned_label4 = [measure[5] for measure in unpruned_measures10]
    # pruned_label4 = [measure[5] for measure in pruned_measures90]


##############################################################################
                            # AVERAGES #

    av_unpruned_maxDepth = averageList(unpruned_maxDepth)
    av_pruned_maxDepth = averageList(pruned_maxDepth)

    av_unpruned_classificationRate, \
    av_unpruned_confusionMatrix, \
    av_unpruned_depth, \
    min_unpruned_depth, \
    max_unpruned_depth, \
    unpruned_room_stats = get_avg_stats(unpruned_measures10)


    av_pruned_classificationRate, \
    av_pruned_confusionMatrix, \
    av_pruned_depth, \
    min_pruned_depth, \
    max_pruned_depth, \
    pruned_room_stats = get_avg_stats(pruned_measures90)


###############################################################################
                         # PLOT THE RESULTS #

    # Plot unpruned distributions

    # Classification rate
    plot_histogram([unpruned_classificationRate, pruned_classificationRate], 2, 'Classification distribution\nUnpruned vs. Pruned\n{}'.format(data_name))
    # plot_histogram([pruned_classificationRate], 9, 'Pruned classification distribution across test sets', plotting90measures=True)

    # Max depth
    plot_histogram([unpruned_maxDepth, pruned_maxDepth], 2, 'Maximum depth distribution\nUnpruned vs. Pruned\n{}'.format(data_name))
    # plot_histogram([pruned_maxDepth,], 9, 'Pruned maximum depth distribution', plotting90measures=True)



##############################################################################
                            # PRINT RESULTS #

    print('===========================================================================')
    print('======================== UNPRUNED ({}) ========================='.format(data_name))
    print('\n')
    print('1. Average classification accuracy: {}%'.format(av_unpruned_classificationRate))
    print('2. Average max depth: {} layers'.format(av_unpruned_maxDepth))
    print('3. Average confusion matrix: \n {}\n'.format(av_unpruned_confusionMatrix))
    print('4. Label-specific stats:')
    for i in unpruned_room_stats:
        print(i, end='\n')
    print('===========================================================================')

    print('===========================================================================')
    print('======================== PRUNED ({}) ==========================='.format(data_name))
    print('\n')
    print('1. Average classification accuracy: {}%\n'.format(av_pruned_classificationRate))
    print('2. Average max depth: {} layers\n'.format(av_pruned_maxDepth))
    print('3. Average confusion matrix: \n {}\n'.format(av_pruned_confusionMatrix))
    print('4. Label-specific stats:')
    for i in pruned_room_stats:
        print(i)
    print('===========================================================================')


if __name__ == '__main__':

    lst = np.array([1,2,3,2,3,3,3,3,3,4,4,5,4,5,6,6,7,6,5,5,4,3,3])

    plot_histogram(lst, 2, 'Classification distribution', normalising=True)


