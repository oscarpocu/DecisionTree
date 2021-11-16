import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from DecisionTree import  DecisionTree

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
    decision_tree = DecisionTree()
    decision_tree.fit(X,Y)
    a = 3


if __name__ == "__main__":
    main()