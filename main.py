import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from CrossValidation import *
from DecisionTree import  DecisionTree

INPATH = 'data/'


def write_out_tree(out, filename='data/out.txt'):
    f = open(filename, "w")
    f.write(out)
    f.close()


def continuos_to_discrete_attr(data, index, n=2):
    for i in index:
        data[:, i] = pd.qcut(data[:, i], n, duplicates='drop')
    return data


def standard_norm(data, index):
    st = StandardScaler()
    data[:, index] = st.fit_transform(data[:, index])
    return data, st


def load_dataset(path):
    dataset = pd.read_csv(path, header=0, delimiter=',', skipinitialspace = True)
    return dataset


def main():
    dataset = load_dataset(INPATH+'adult.data')
    dataset = dataset.replace({'?': np.NaN})
    dataset = dataset.dropna()

    # todo: automatitzar-ho amb np.unique()

    data = dataset.to_numpy()

    data = continuos_to_discrete_attr(data, [0, 2, 4, 10, 11, 12], n=6)
    X = data[:, :-1]
    Y = data[:, -1]

    print(X.shape, data.shape)
    decision_tree = DecisionTree(criterion="entropy_ratio")
    decision_tree.fit(X, Y, dataset.columns[:-1])
    #write_out_tree(str(decision_tree))
    print(decision_tree)
    print("\n\n--------------------Test with Training set--------------------\n")
    for i in range(min(dataset.shape[0], 20)):
        x = X[i, :]
        ground_truth = Y[i]
        predict = decision_tree.predict(x)
        print("Index ["+str(i)+"]; Prediction: "+str(predict)+" GT: "+str(ground_truth)+"     "+("Correct" if ground_truth==predict else "Incorrect"))

    print("\n\nCross Validation Score: ", cross_val_score(decision_tree, X, Y, scoring="accuracy"))


# todo: convert continuous attributes
if __name__ == "__main__":
    main()