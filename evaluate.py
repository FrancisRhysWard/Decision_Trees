import numpy as np
from random import choice, randint

clean_dataset = np.loadtxt("./co553-cbc-dt/wifi_db/clean_dataset.txt")

##noisy_data = np.loadtxt("./co553-cbc-dt/wifi_db/noisy_dataset.txt")




def divide_data(data):

    np.random.shuffle(data) ## shuffles data

    return np.split(data, 10)  ## returns list of 10 np arrays used as CV folds


def cross_validation(data):

    divided_data = divide_data(data)

    for i in range(10):
        test_data = divided_data[i]

        validation_data = divided_data[(i+1) % 10]

        training_data = np.array(divided_data[(i+2) % 10: i]).flatten()

if __name__ == "__main__":
    cross_validation(clean_dataset)


