import numpy as np


def gini(x, x_classes):
    g_sum = 0
    for v in x_classes:
        condition = (v == x)
        product = (x[condition].shape[0] / x.shape[0]) ** 2
        g_sum += product
    return 1 - g_sum


def gini_gain(x, y, x_classes, y_classes):
    g_sum = 0
    for y_class in y_classes:
        condition = (y == y_class)
        product = x[condition].shape[0] / x.shape[0]
        product *= gini(x[condition], x_classes)
        g_sum += product
    return gini(x, x_classes) - g_sum


def entropy(x, x_classes, y=None, y_classes=None):
    sum = 0
    if y is None or y_classes is None:
        for x_class in x_classes:
            prob = np.sum(x == x_class) / x.shape[0]
            sum += (prob * np.log2(prob)) if prob != 0 else 0
        return -sum
    else:
        for y_class in y_classes:
            x_cond = x[y_class == y]
            sum += ((x_cond.shape[0] / x.shape[0]) * entropy(x_cond, x_classes))
        return sum


def entropy_gain(x, y, x_classes, y_classes):
    return entropy(x, x_classes) - entropy(x, x_classes, y, y_classes)


def split_info(x, y, x_classes, y_classes):
    sum = 0
    for y_class in y_classes:
        prop = (x[y_class == y].shape[0]/x.shape[0])
        sum += prop * np.log2(prop)
    return -sum


def gain_ratio_entropy(x, y, x_classes, y_classes):
    return entropy_gain(x, y, x_classes, y_classes) / split_info(x, y, x_classes, y_classes)


def gain_ratio_gini(x, y, x_classes, y_classes):
    return gini_gain(x, y, x_classes, y_classes) / split_info(x, y, x_classes, y_classes)