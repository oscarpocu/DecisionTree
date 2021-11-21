from Metrics import *
import time

from Node import Node


class DecisionTree():
    def __init__(self, criterion='gini'):
        if criterion == 'gini':
            self.criterion = gini_gain
        elif criterion == 'entropy':
            self.criterion = entropy_gain
        self.attribute_names = None
        self.attribute_classes = []  # different classes in every attribute
        self.prediction_classes = []
        self.used_attributes = []
        self.x = None
        self.y = None
        self.used_attributes = None
        self.root = None
        self.max_time = 60
        self.init_time = -1

    def rec_fit(self, node, x, y):
        # if all predictions are the same we have the decision
        if np.unique(y).shape[0] == 1:
            node.predictions[y[0]] = 1
            return

        # if we have different possible predictions then we use probabilities
        for p_class in np.unique(y):
            node.predictions[p_class] = y[(p_class == y)].shape[0]/y.shape[0]

        # if max time is reached then fit is ended
        if self.max_time <= time.time() - self.init_time:
            return

        # no more attributes to keep training
        if x.shape[0] == 0 or x.shape[1] == 0:
            return

        # calculate the best attribute
        evaluation_metrics = []
        for i in range(x.shape[1]):
            evaluation_metrics.append(self.criterion(x[:, i], y, self.attribute_classes[i], self.prediction_classes))
        attribute_index = np.argmax(np.array(evaluation_metrics))
        node.attribute_index = attribute_index

        # create child nodes
        for a_class in np.unique(x[:, attribute_index]):
            prediction_classes = np.unique(y[x[:, attribute_index] == a_class])
            node.add_child((a_class, Node(node, prediction_classes, attribute_index)))

        # call recursive fit with new x and y
        indices = np.arange(x.shape[1])
        for child_i in range(len(node.childs)):
            valid_rows = x[:, attribute_index] == node.childs[child_i][0]
            valid_cols = indices != attribute_index
            new_x = x[valid_rows, :]
            new_x = new_x[:, valid_cols]
            new_y = y[valid_rows]
            self.rec_fit(node.childs[child_i][1], new_x, new_y)

        return

    def fit(self, x, y, attribute_names=None):
        self.init_time = time.time()
        if attribute_names is not None and x.shape[1] == len(attribute_names):
            self.attribute_names = attribute_names
        else:
            self.attribute_names = []
            for j in range(x.shape[1]):
                self.attribute_names.append("Attribute"+str(j))
        self.x = x
        self.y = y
        self.prediction_classes = np.unique(y)
        self.root = Node(None, self.prediction_classes)
        for j in range(x.shape[1]):
            self.attribute_classes.append(np.unique(x[:, j]))
        self.rec_fit(self.root, self.x, self.y)

    def predict(self, x):
        pass

    def prune(self):
        pass

    def rec_str(self, out, node, level):
        level += 1
        if len(node.childs) == 0:
            out += " --> "
            max_key = max(node.predictions, key=node.predictions.get)
            out += max_key

        for child in node.childs:
            out += "\n"
            out += (level*"-------")
            out += "|"+str(child[0])
            out = self.rec_str(out, child[1], level)
        level -= 1
        return out

    def __str__(self):
        out = "\n"
        out += "Root"
        return self.rec_str(out, self.root, 0)






