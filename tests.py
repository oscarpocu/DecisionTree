import unittest

import numpy as np
from math import trunc
import main
from DecisionTree import DecisionTree


class DecisionTreeTests(unittest.TestCase):
    def test_entropy(self):
        decision_tree = DecisionTree()
        dataset = main.load_dataset("data/test.data")
        data = dataset.to_numpy()
        x = data[:, -3]
        y = data[:, -1]
        res = decision_tree.entropy_gain(x, np.unique(x), y, np.unique(y))
        self.assertEqual(0.151*1000, trunc(res*1000))


if __name__ == '__main__':
    unittest.main()
