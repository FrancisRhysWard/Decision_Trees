import numpy as np
from random import choice, randint

clean_dataset = np.loadtxt("./wifi_db/clean_dataset.txt")

data_size = len(clean_dataset)

wifi_attr = [i for i in range(1, len(clean_dataset[0]))]


def p_k(_label, _dataset):
    '''
    Calculates the sample probability of _label in _dataset as a float
    :param _label: label
    :param _dataset: dataset
    :return: probability (float)
    '''

    S_size = len(_dataset)

    if S_size == 0:
       return 1  ## entropy 0 and therefore negative gain

    number_k_in_S = len([s for s in _dataset if s[-1] == _label]) ## number of samples in S with label k

    return number_k_in_S / S_size


def H(_dataset):
    '''
    Calculates entropy of a dataset

    :param _dataset: dataset
    :return: entropy value (float)
    '''

    labels = [1,2,3,4]
    list_of_p = [p_k(k, _dataset) for k in labels if p_k(k, _dataset) != 0]  ## create ps so they only need to be calcd once
    summands = [p * np.log2(p) for p in list_of_p]                           ## create H summands

    return - sum(summands)


def split_by_set(_dataset, wifi_index, value):
    '''
    Splits by one of the unique values from the set of that particular wifi attribute
    :param _dataset: dataset to be split
    :param wifi_index: split attribute
    :param value: split attribute value
    :return: dataset_left, dataset_right,               --> children's datasets
             split attribute, split attribute value
    '''

    dataset_left = np.array([sample for sample in _dataset if sample[wifi_index] <= value])

    dataset_right = np.array([sample for sample in _dataset if sample[wifi_index] > value])

    return dataset_left, dataset_right, wifi_attr[wifi_index], value


def remainder(split):
    '''
    Calculate remainder
    :param split: all the information about the split (including datasets)
    :return: remainder
    '''

    dataset_left, dataset_right = split[:2]
    size_left = len(dataset_left)
    size_right = len(dataset_right)

    return (size_left / (size_left + size_right) * H(dataset_left)) + (size_right / (size_left + size_right) * H(dataset_right))


def Gain(_dataset, split):
    '''
    Calculates gain
    :param _dataset:
    :param split: all the information about the split (including datasets)
    :return:
    '''
    return H(_dataset) - remainder(split)



def find_split(dataset):
    '''
    Finds the split
    :param dataset:
    :return: max_gain: gain of the split
             splits[max_index]: dataset_left, dataset_right,
             wifi_attr[wifi_index], mean_split_value,          --> i.e. Left data, Right data, Attribute, Attribute value
    '''

    if len(dataset) == 0:
        return None

    splits = []
    gains_for_each_attribute = []

    # Iterate through all wifi attributes
    for wifi_idx in range(len(wifi_attr)):
        # print('Checking wifi{}'.format(wifi_idx))

        # These are all unique values for that wifi attribute
        values = set([sample[wifi_idx] for sample in dataset])

        gains_for_each_value = []
        splits_by_value = []
        # Now we split for each value -> calculate max Gain
        for value in values:

            # print('Checking value: {}'.format(value))
            split = split_by_set(dataset, wifi_idx, value)
            splits_by_value.append(split)

            # Now calculate the gain from this split and add to gains list
            gains_for_each_value.append(Gain(dataset, split))

            # Choose index of max gain
            max_gain_idx = np.argmax(gains_for_each_value)

            max_gain = gains_for_each_value[max_gain_idx]

        # Add split corresponding to max gain for a particular wifi attribute
        splits.append(splits_by_value[max_gain_idx])
        gains_for_each_attribute.append(max_gain)


    max_index = np.argmax(gains_for_each_attribute)
    max_gain = gains_for_each_attribute[max_index]

    # If we can't find a good split
    if max_gain <= 0:
        return (None, (None, None, None, None))
    elif np.array([]) in splits[max_index][:2]:
        return (None, (None, None, None, None))


    return max_gain, splits[max_index]



if __name__ == "__main__":

    # Sandbox

    print(H(clean_dataset))

    print(f'type clean data = {type(clean_dataset)}', clean_dataset[:10])


    good_split = find_split(clean_dataset)
    gain, split_result = good_split
    sl, sr, wifi, split_value = split_result


    print(f"Total entropy = {H(clean_dataset)}, Gain = {gain} \n wfi = {wifi} \n split value = {split_value}")

    split_left = find_split(sl)
    gain, split_result = split_left
    sl2, sr, wifi, split_value = split_result

    print(f"S_left entropy = {H(sl)}, Gain = {gain} \n wfi = {wifi} \n split value =       {split_value}")

    zero_size_arr = np.array([])
    print(f"find split of zero size array outputs: {find_split(zero_size_arr)}")
    '''
    split = split(clean_dataset)
    S_left, S_right, wifi, split_value = split
    print(f'entropy of s_left = {H(S_left)}, entropy of s_right = {H(S_right)} \n wifi signal = {wifi}, split value = {split_value}')
    
    print(f'Gain = {Gain(clean_dataset,split )}')
    '''
