import numpy as np

from Node import Node


class DecisionTree():
    def __init__(self, criterion='gini'):
        self.criterion = criterion
        self.classes = []
        self.root = Node(None, None)
        self.used_attributes = []
        self.x = None
        self.y = None

    def rec_fit(self, node):
        for i, y in enumerate(self.attribute_classes):
            gni = self.gini_coefficient( self.x[:, i], self.attribute_classes[i])
            # Formulita GiniGain
        pass

    def fit(self, x, y):
        self.classes = np.unique(y)

    def predict(self, x):
        pass

    def prune(self):
        pass
