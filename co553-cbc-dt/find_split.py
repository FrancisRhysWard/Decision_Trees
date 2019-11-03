import numpy as np
from random import choice, randint

clean_dataset = np.loadtxt("./wifi_db/clean_dataset.txt")

data_size = len(clean_dataset)

wifi_attr = [i for i in range(1, len(clean_dataset[0]))]


def p_k(k, S):
    '''
    takes a label k and array of data S and returns the sample probability of k in S as a float
    '''

    S_size = len(S)

    if S_size == 0:
       return 1  ## entropy 0 and therefore negative gain

    number_k_in_S = len([s for s in S if s[-1] == k]) ## number of samples in S with label k

    return number_k_in_S / S_size


def H(S):
    '''
    Entropy of a dataset S
    returns float
    '''

    labels = [1,2,3,4]
    list_of_p = [p_k(k, S) for k in labels if p_k(k, S) != 0] ## create ps so they only need to be calcd once
    summands = [p * np.log2(p) for p in list_of_p]  ## create H summands
    return - sum(summands)


def split_by_set(S, wifi_index, value):
    '''
    Splits by one of the unique values from the set of that particular wifi attribute
    :param S:
    :param wifi_index:
    :param value:
    :return:
    '''
    S_left = np.array([sample for sample in S if sample[wifi_index] <= value])

    S_right = np.array([sample for sample in S if sample[wifi_index] > value])

    return S_left, S_right, wifi_attr[wifi_index], value




def remainder(split):
    S_left, S_right = split[:2]
    size_left = len(S_left)
    size_right = len(S_right)

    return (size_left / (size_left + size_right) * H(S_left)) + (size_right / (size_left + size_right) * H(S_right))

## print(remainder(split(clean_dataset)))

def Gain(S, split):
    return H(S) - remainder(split)



def find_split(S):
    '''

    :param S:
    :return: max_gain: gain of the split
             splits[max_index]: S_left, S_right, wifi_attr[wifi_index], mean_split_value, i.e. Left data, Right data, Attribute, Attribute value
    '''
    if len(S) == 0:
        return None

    splits = []
    gains_for_each_attribute = []

    # Iterate through all wifi attributes
    for wifi_idx in range(len(wifi_attr)):
        # print('Checking wifi{}'.format(wifi_idx))

        # These are all unique values for that wifi attribute
        values = set([sample[wifi_idx] for sample in S])

        gains_for_each_value = []
        splits_by_value = []
        # Now we split for each value -> calculate max Gain
        for value in values:

            # print('Checking value: {}'.format(value))
            split = split_by_set(S, wifi_idx, value)
            splits_by_value.append(split)

            # Now calculate the gain from this split and add to gains list
            gains_for_each_value.append(Gain(S, split))

            # Choose index of max gain
            max_gain_idx = np.argmax(gains_for_each_value)

            max_gain = gains_for_each_value[max_gain_idx]

        # Add split corresponding to max gain for a particular wifi attribute
        splits.append(splits_by_value[max_gain_idx])
        gains_for_each_attribute.append(max_gain)


    max_index = np.argmax(gains_for_each_attribute)
    max_gain = gains_for_each_attribute[max_index]


    if max_gain <= 0:
        return (None, (None, None, None, None))
    elif np.array([]) in splits[max_index][:2]:
        return (None, (None, None, None, None))


    return max_gain, splits[max_index]



if __name__ == "__main__":

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
