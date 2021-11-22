import numpy as np


class Node:
    def __init__(self, parent, prediction_classes, attribute_name=None):
        self.childs = []  # [(Class, Child), ]
        self.parent = parent
        self.attribute_name = attribute_name  # Node Attribute (x column)
        self.predictions = {}  # Amount of samples of each class
        for p_class in prediction_classes:
            self.predictions[p_class] = 0

    def add_child(self, node):
        self.childs.append(node)

    def set_attribute_name(self, attribute_name):
        self.attribute_name = attribute_name
