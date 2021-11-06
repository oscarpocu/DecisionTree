import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

INPATH = 'data/'


def load_dataset(path):
    dataset = pd.read_csv(path, header=0, delimiter=',')
    return dataset


def main():
    dataset = load_dataset(INPATH+'test.data')
    data = dataset.to_numpy()
    X = data[:, :-1]
    Y = data[:, -1]
    print(X.shape, data.shape)
    print(dataset.head())
    print(dataset.dtypes)





if __name__ == "__main__":
    main()