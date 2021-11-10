import numpy as np

from Node import Node


class DecisionTree():
    def __init__(self, criterion='gini'):
        self.criterion = criterion
        self.attribute_classes = []
        self.prediction_classes = []
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
        self.x = x
        self.y = y
        self.prediction_classes = np.unique(y)
        for j in range(x.shape[1]):
            self.attribute_classes.append(np.unique(x[:, j]))
        self.rec_fit(self.root)

    def predict(self, x):
        pass

    def prune(self):
        pass

    # s -> conjunt delements duna columna, y -> classes possibles
    def gini_coefficient(self, x, x_classes):
        g_sum = 0
        for v in x_classes:
            condition = (v == x)
            product = (x[condition].shape[0]/x.shape[0])**2
            g_sum += product
        return 1 - g_sum

    def gini_gain(self, x, y, x_classes, y_classes):
        g_sum = 0
        for y_class in y_classes:
            condition = (y == y_class)
            product = x[condition].shape[0]/x.shape[0]
            product *= self.gini_coefficient(x[condition], x_classes)
            g_sum += product
        return self.gini_coefficient(x, x_classes) - g_sum


