import numpy as np


class Node:
    def __init__(self, parent, attribute=None, classes=None):
        self.childs = []  # [(Class, Child), ]
        self.posible_values = []
        self.parent = parent
        self.attribute = attribute  # Node Attribute
        self.decision = {}  # Amount of samples of each class
