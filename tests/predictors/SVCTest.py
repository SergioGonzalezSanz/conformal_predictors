import unittest
from conformal_predictors.nc_measures.SVM import SVCDistanceNCMeasure
from conformal_predictors.predictors.SVC import ConformalSVC
from numpy import array


class SVMTest(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_1(self):
        x = array([[1, 1], [2, 2], [3, 3]])
        y = array([0, 1, 2])
        measure = SVCDistanceNCMeasure()
        cp = ConformalSVC(measure)
        cp.fit(x, y)
        cp.calibrate(x, y)
        cp.predict_cf(x)
        predicted_l, credibility, confidence = cp.predict_cf(x)
        self.assertEqual(0, predicted_l[0])
        self.assertEqual(1, predicted_l[1])
        self.assertEqual(2, predicted_l[2])
        self.assertAlmostEqual(0.66666667, credibility[0])
        self.assertAlmostEqual(1, credibility[1])
        self.assertAlmostEqual(0.66666667, credibility[2])
        self.assertEqual(1, confidence[0])
        self.assertEqual(1, confidence[1])
        self.assertEqual(1, confidence[2])

    def test_1(self):
        x = array([[1, 1], [2, 2], [3, 3], [4, 4]])
        y = array([0, 1, 2, 3])
        measure = SVCDistanceNCMeasure()
        cp = ConformalSVC(measure)
        cp.fit(x, y)
        cp.calibrate(x, y)
        cp.predict_cf(x)
        predicted_l, credibility, confidence = cp.predict_cf(x)
        # self.assertEqual(0, predicted_l[0])
        # self.assertEqual(1, predicted_l[1])
        # self.assertEqual(2, predicted_l[2])
        # self.assertAlmostEqual(0.66666667, credibility[0])
        # self.assertAlmostEqual(1, credibility[1])
        # self.assertAlmostEqual(0.66666667, credibility[2])
        # self.assertEqual(1, confidence[0])
        # self.assertEqual(1, confidence[1])
        # self.assertEqual(1, confidence[2])
        print(cp._clf.predict(x))
        print(cp._nc_measure.evaluate(cp._clf, x))
        print(predicted_l)
        print(credibility)
        print(confidence)