import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from DecisionTree import  DecisionTree

INPATH = 'data/'


def write_out_tree(out, filename='data/out.txt'):
    f = open(filename, "w")
    f.write(out)
    f.close()


def process_gain_and_loss(data):
    data[:, 0] = np.where(data[:, 0] == 0, 'capital-gain == 0', data[:, 0])
    data[:, 0] = np.where(data[:, 0] != 'capital-gain == 0', 'capital-gain > 0', data[:, 0])

    data[:, 1] = np.where(data[:, 1] == 0, 'capital-loss == 0', data[:, 1])
    data[:, 1] = np.where(data[:, 1] != 'capital-loss == 0', 'capital-loss > 0', data[:, 1])

    return data


def continuos_to_discrete_attr(data, index, n=2):
    for i in index:
        data[:, i] = pd.qcut(data[:, i], n)
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
    dataset = dataset.replace({' ?': np.NaN})
    dataset = dataset.dropna()

    # todo: automatitzar-ho amb np.unique()

    data = dataset.to_numpy()

    data[:, [10, 11]] = process_gain_and_loss(data[:, [10, 11]])
    data = continuos_to_discrete_attr(data, [0, 2, 4, 12])
    X = data[:, :-1]
    Y = data[:, -1]

    print(X.shape, data.shape)
    decision_tree = DecisionTree(criterion="gini_ratio")
    decision_tree.fit(X, Y, dataset.columns[:-1])
    #write_out_tree(str(decision_tree))
    print(decision_tree)
    print("\n\n--------------------Test with Training set--------------------\n")
    for i in range(min(dataset.shape[0], 20)):
        x = X[i,:]
        ground_truth = Y[i]
        predict = decision_tree.predict(x)
        print("Index ["+str(i)+"]; Prediction: "+str(predict)+" GT: "+str(ground_truth)+"     "+("Correct" if ground_truth==predict else "Incorrect"))

# todo: convert continuous attributes
if __name__ == "__main__":
    main()