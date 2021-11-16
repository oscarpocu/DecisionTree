import numpy as np


class Node:
    def __init__(self, parent,  prediction_classes, attribute=None):
        self.childs = []  # [(Class, Child), ]
        self.parent = parent
        self.attribute = attribute  # Node Attribute
        self.predictions = {}  # Amount of samples of each class
        for p_class in prediction_classes:
            self.predictions[p_class] = 0

    def add_child(self, node):
        self.childs.append(node)
