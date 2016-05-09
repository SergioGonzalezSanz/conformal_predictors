import unittest
from conformal_predictors.nc_measures.SVM import SVCDistanceNCMeasure
from sklearn.svm import SVC
from numpy import array


class SVMTest(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_1(self):
        x = array([[1, 1], [2, 2]])
        y = array([0, 1])
        measure = SVCDistanceNCMeasure()
        clf = SVC(decision_function_shape='ovr')
        clf.fit(x, y)
        measures = measure.evaluate(clf, x)
        self.assertAlmostEqual(measures[0, 0], -.63212056)
        self.assertAlmostEqual(measures[0, 1], .63212056)
        self.assertAlmostEqual(measures[1, 0], .63212056)
        self.assertAlmostEqual(measures[1, 1], -.63212056)

    def tests_2(self):
        x = array([[1, 1], [2, 2], [3, 3]])
        y = array([0, 1, 2])
        measure = SVCDistanceNCMeasure()
        clf = SVC(decision_function_shape='ovr')
        clf.fit(x, y)
        measures = measure.evaluate(clf, x)
        self.assertAlmostEqual(measures[0, 0], -1.5)
        self.assertAlmostEqual(measures[0, 1], 1.08754365)
        self.assertAlmostEqual(measures[0, 2], .41245635)
        self.assertAlmostEqual(measures[1, 0], 1.19584788)
        self.assertAlmostEqual(measures[1, 1], -1.60830423)
        self.assertAlmostEqual(measures[1, 2], .19584788)
        self.assertAlmostEqual(measures[2, 0], .41245635)
        self.assertAlmostEqual(measures[2, 1], 1.08754365)
        self.assertAlmostEqual(measures[2, 2], -1.5)
