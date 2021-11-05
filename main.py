import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

INPATH = 'data/'


def load_dataset(path):
    dataset = pd.read_csv(path, header=0, delimiter=',')
    return dataset


def main():
    dataset = load_dataset(INPATH+'adult.data')
    print(dataset.head())





if __name__ == "__main__":
    main()