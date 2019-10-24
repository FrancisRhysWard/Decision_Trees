import numpy as np
from random import choice, randint

clean_dataset = np.loadtxt("./co553-cbc-dt/wifi_db/clean_dataset.txt")

data_size = len(clean_dataset)

wifi_attr = [i for i in range(1, len(clean_dataset[0]))]

print(clean_dataset[:10])

def p_k(k, S):
    '''
    takes a label k and array of data S and returns the sample probability of k in S as a float
    '''

    S_size = len(S)
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

def split(S):
    random_attribute_index = choice(range(len(S[0]) - 1)) ## chooses a random wifi signal
   ## rand_data_subset = clean_dataset[randint(0, 1000): randint(1000, -1)]

    size = len(S)

    rand_data_subset = S[int(size * 0.2): int(size * 0.8)]

    values = [sample[random_attribute_index] for sample in rand_data_subset]

    rand_split_value = choice(values)

    S_left = [sample for sample in S if sample[random_attribute_index] <= rand_split_value]

    S_right = [sample for sample in S if sample[random_attribute_index] > rand_split_value]
    return S_left, S_right, wifi_attr[random_attribute_index], rand_split_value


def remainder(S_l_r):
    S_left, S_right = S_l_r[:2]
    size_left = len(S_left[0])
    size_right = len(S_right)

    return (size_left / (size_left + size_right) * H(S_left)) + (size_right / (size_left + size_right) * H(S_right))

## print(remainder(split(clean_dataset)))

def Gain(S, split):
    return H(S) - remainder(split)



def find_split(S):
    splits = [split(S) for i in range(50)]
    gains = [Gain(S, split) for split in splits]
    max_index = np.argmax(gains)
    return gains[max_index], splits[max_index]


if __name__ == "__main__":

    good_split = find_split(clean_dataset)

    gain, split = good_split
    sl, sr, wifi, split_value = split

    print(f"Gain = {gain} \n wfi = {wifi} \n split value = {split_value}")

'''
split = split(clean_dataset)
S_left, S_right, wifi, split_value = split
print(f'entropy of s_left = {H(S_left)}, entropy of s_right = {H(S_right)} \n wifi signal = {wifi}, split value = {split_value}')

print(f'Gain = {Gain(clean_dataset,split )}')
'''
