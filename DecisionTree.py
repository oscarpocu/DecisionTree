import numpy as np

from Node import Node


class DecisionTree():
    def __init__(self, criterion='gini'):
        self.criterion = criterion
        self.attribute_classes = []
        self.prediction_classes = []
        self.used_attributes = []
        self.x = None
        self.y = None
        self.used_attributes = None
        self.root = None

    def rec_fit(self, node, x, y):
        # if all predictions are the same we have the decision
        if np.unique(y).shape[0] == 1:
            node.predictions[y[0]] = 1
            return

        # if we have different possible predictions then we use probabilities
        for p_class in np.unique(y):
            node.predictions[p_class] = (p_class == y).shape[0]/y.shape[0]

        # calculate the best attribute
        gini_gains = []
        for i in range(len(self.attribute_classes)):
            gini_gains.append(self.gini_gain(x[:, i], y, self.attribute_classes[i], self.prediction_classes))
        attribute_index = np.argmax(np.array(gini_gains))
        node.attribute = self.attribute_classes[attribute_index]

        # create child nodes
        for a_class in np.unique(x[:, attribute_index]):
            prediction_classes = np.unique(y[x[:, attribute_index] == a_class])
            node.add_child((a_class, Node(node, prediction_classes)))

        # call recursive fit with new x and y
        indices = np.arange(x.shape[0])
        for child in node.childs:
            new_x = x[x[:, attribute_index] == child[0], indices != attribute_index]
            new_y = y[x[:, attribute_index] == child[0]]
            self.rec_fit(child[1], new_x, new_y)

        return

    def fit(self, x, y):
        self.x = x
        self.y = y
        self.prediction_classes = np.unique(y)
        self.root = Node(None, np.zeros(x.shape[1]), self.prediction_classes)
        for j in range(x.shape[1]):
            self.attribute_classes.append(np.unique(x[:, j]))
        self.rec_fit(self.root, self.x, self.y)

    def predict(self, x):
        pass

    def prune(self):
        pass

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


