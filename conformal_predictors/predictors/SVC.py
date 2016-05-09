from conformal_predictors.nc_measures import NCMeasure
from sklearn.svm import SVC, NuSVC
from conformal_predictors.predictors import ConformalPredictor
from abc import ABCMeta, abstractmethod
import numpy as np


class ConformalSVM(ConformalPredictor):

    _metaclass_ = ABCMeta

    def __init__(self, nc_measure, clf):
        super().__init__(nc_measure, clf)

    def fit(self, x: np.ndarray, y: np.ndarray, sample_weight: np.array = np.empty(0)) -> None:
        self._clf.fit(x, y, sample_weight)

    def calibrate(self, cal: np.ndarray, cal_l: np.ndarray) -> None:
        self._cal_l = cal_l

        # Alpha values of each calibration point for each class
        alphas = self._nc_measure.evaluate(self._clf, cal)

        # Indices of the class in cal_l
        dict = self._clf.classes_.tolist()
        l_pos = [dict.index(val) for val in cal_l]

        # The alpha is chosen accordingly to the calibration label
        self._cal_a = [alphas[i, l_pos[i]] for i in range(0, cal.shape[0])]

    def predict_cf(self, x: np.ndarray, **kwargs):
        self.check_calibrated()
        dict = self._clf.classes_.tolist()  # Classes labels
        alphas = self._nc_measure.evaluate(self._clf, x)
        # print(alphas)
        predicted_l = np.array([0.0] * x.shape[0])
        confidence = np.array([0.0] * x.shape[0])
        credibility = np.array([0.0] * x.shape[0])
        for i in range(0, alphas.shape[0]):
            print(alphas[i, :])
            p_values = self.compute_pvalue(alphas[i])
            conformal = np.sort(p_values)
            predicted_l[i] = dict[np.argmax(p_values)]
            credibility[i] = conformal[-1]
            confidence[i] = 1.0 - conformal[-2]
            print(p_values)
            print(predicted_l[i])
            print(credibility[i])
            print(confidence[i])

        return [predicted_l, credibility, confidence]


class ConformalSVC(ConformalSVM):

    def __init__(self, nc_measure: NCMeasure, C=1.0, kernel='rbf', degree=3, gamma='auto',
                 coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200,
                 class_weight=None, verbose=False, max_iter=-1, random_state=None):
        super().__init__(nc_measure, SVC(C=C, kernel=kernel, degree=degree, gamma=gamma,
                                         coef0=coef0, shrinking=shrinking, probability=probability,
                                         tol=tol, cache_size=cache_size, class_weight=class_weight,
                                         verbose=verbose, max_iter=max_iter,
                                         decision_function_shape='ovr', random_state=random_state))


class ConformalNuSVC(ConformalSVM):

    def __init__(self, nc_measure: NCMeasure, nu=0.5, kernel='rbf', degree=3, gamma='auto',
                 coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200,
                 class_weight=None, verbose=False, max_iter=-1, random_state=None):
        super().__init__(nc_measure, NuSVC(nu=nu, kernel=kernel, degree=degree, gamma=gamma,
                                           coef0=coef0, shrinking=shrinking,
                                           probability=probability, tol=tol, cache_size=cache_size,
                                           class_weight=class_weight, verbose=verbose,
                                           max_iter=max_iter, decision_function_shape='ovr',
                                           random_state=random_state))


