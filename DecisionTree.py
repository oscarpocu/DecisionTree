import numpy as np

from Node import Node


class DecisionTree():
    def __init__(self, criterion='gini'):
        self.criterion = criterion
        self.classes = []
        self.root = Node(None, None)

    def fit(self, x, y):
        self.classes = np.unique(y)

    def predict(self, x):
        pass

    def prune(self):
        pass
