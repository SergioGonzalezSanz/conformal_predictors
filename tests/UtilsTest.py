import unittest
from conformal_predictors.utils import Utils
from numpy import array
from numpy.random import rand


class UtilsTest(unittest.TestCase):

    def test_1(self):
        bag = array([2, 3, 4])
        value = 1
        pvalue = Utils.compute_pvalue(bag, value)
        self.assertEqual(pvalue, 1, "Check 1: max pvalue")

    def test_2(self):
        bag = rand(10, 1)
        value = 1
        pvalue = Utils.compute_pvalue(bag, value)
        self.assertEqual(pvalue, 1/(len(bag) + 1.0), "Check 2: min pvalue, 10-element-bag")

    def test_3(self):
        bag = rand(1000, 1)
        value = 1
        pvalue = Utils.compute_pvalue(bag, value)
        self.assertEqual(pvalue, 1/(len(bag) + 1.0), "Check 3: min pvalue, 1000-element-bag")

    def test_4(self):
        bag = rand(1000, 1)
        value = -1
        pvalue = Utils.compute_pvalue(bag, value)
        self.assertEqual(pvalue, 1, "Check 4: max value, 1000-element-bag")

if __name__ == '__main__':
    unittest.main()
