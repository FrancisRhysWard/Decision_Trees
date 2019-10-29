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

## print(p_k(1.0, clean_dataset[:10]))

def H(S):
    '''
    Entropy of a dataset S
    returns float
    '''

    labels = [1,2,3,4]
    list_of_p = [p_k(k, S) for k in labels if p_k(k, S) != 0] ## create ps so they only need to be calcd once
    summands = [p * np.log2(p) for p in list_of_p]  ## create H summands
    return - sum(summands)

## print(H(clean_dataset))

def split(S, wifi_index):

        size = len(S)

        ## random_attribute_index = choice(range(len(S[0]) - 1)) ## chooses a random wifi signal

        values = [sample[wifi_index] for sample in S]

        mean_split_value = sum(values) / size

        ##rand_split_value = choice(values)

        ##print(f"wifi {wifi_attr[wifi_index]} split at mean value {mean_split_value}")

        S_left = np.array([sample for sample in S if sample[wifi_index] <= mean_split_value])

        S_right = np.array([sample for sample in S if sample[wifi_index] > mean_split_value])

        return S_left, S_right, wifi_attr[wifi_index], mean_split_value


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

    for i in range(len(wifi_attr)):
        result_from_split = split(S, i)
        splits.append(result_from_split)
    gains = [Gain(S, split) for split in splits]
    max_index = np.argmax(gains)
    max_gain = gains[max_index]


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
