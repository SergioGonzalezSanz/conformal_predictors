from conformal_predictors.nc_measures import NCMeasure
from sklearn.svm import SVC, NuSVC
from conformal_predictors.predictors import ConformalPredictor
from abc import ABCMeta, abstractmethod
import numpy as np
import math


class ConformalSVM(ConformalPredictor):

    _metaclass_ = ABCMeta

    def __init__(self, nc_measure, clf):
        super().__init__(nc_measure, clf)

    def fit(self, x: np.ndarray, y: np.ndarray, sample_weight: np.array = np.empty(0)) -> None:
        self._clf.fit(x, y, sample_weight)

    def calibrate(self, cal: np.ndarray, cal_l: np.ndarray) -> None:

        # Dimensions of the calibration set: (n_samples,n_classes)
        self._cal_l = cal_l

        # Alpha values of each calibration point for each class
        self._cal_a = self._nc_measure.evaluate(self._clf, cal)
        dict = self._clf.classes_.tolist()
        multipliers = np.array([1 if dict.index(cal_l[i]) == j else -1 for i in range(cal.shape[0])
                                for j in range(len(dict))]).reshape((cal.shape[0], len(dict)))
        self._cal_a *= multipliers


        # # Indices of the class in cal_l
        # dict = self._clf.classes_.tolist()
        # l_pos = [dict.index(val) for val in cal_l]
        #
        # # The alpha is chosen accordingly to the calibration label
        # self._cal_a = [alphas[i, l_pos[i]] for i in range(0, cal.shape[0])]

    def predict_cf(self, x: np.ndarray, **kwargs):
        self.check_calibrated()
        dict = self._clf.classes_.tolist()  # Classes labels
        alphas = self._nc_measure.evaluate(self._clf, x)
        predicted_l = np.array([0.0] * x.shape[0])
        confidence = np.array([0.0] * x.shape[0])
        credibility = np.array([0.0] * x.shape[0])
        for i in range(0, alphas.shape[0]):
            p_values = self.compute_pvalue(alphas[i, :])
            conformal = np.sort(p_values)
            predicted_l[i] = dict[np.argmax(p_values)]
            credibility[i] = conformal[-1]
            confidence[i] = 1.0 - conformal[-2]

        return [predicted_l, credibility, confidence]

#
# class MulticlassSVC:
#
#     def __init__(self, C=1.0, kernel='rbf', degree=3, gamma='auto',
#                  coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200,
#                  class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr',
#                  random_state=None):
#         self.classes_ = None
#         self.__clfs = None
#         self.__params = {'C': C, 'kernel': kernel, 'degree': degree, 'gamma': gamma, 'coef0': coef0,
#                          'shrinking': shrinking, 'probability': probability, 'tol': tol,
#                          'cache_size': cache_size, 'class_weight': class_weight, 'verbose': verbose,
#                          'max_iter': max_iter, 'decision_function_shape': decision_function_shape,
#                          'random_state': random_state}
#
#     def fit(self, x, y, sample_weight=None):
#         self.classes_ = np.unique(y)
#         n_classes_ = len(self.classes_)
#         self.__clfs = [SVC(C=self.__params['C'], kernel=self.__params['kernel'],
#                            degree=self.__params['degree'], gamma=self.__params['gamma'],
#                            coef0=self.__params['coef0'], shrinking=self.__params['shrinking'],
#                            probability=self.__params['probability'], tol=self.__params['tol'],
#                            cache_size=self.__params['cache_size'],
#                            class_weight=self.__params['class_weight'],
#                            verbose=self.__params['verbose'], max_iter=self.__params['max_iter'],
#                            decision_function_shape=self.__params['decision_function_shape'],
#                            random_state=self.__params['random_state'])
#                        for i in range(0, n_classes_)]
#         print(len(self.__clfs))
#         for i in range(0,n_classes_):
#             training_labels = [0 if value == self.classes_[i] else 1 for value in y]
#             print(training_labels)
#             self.__clfs[i].fit(x, training_labels, sample_weight=None)
#
#     def decision_function(self, x):
#         return [clf.decision_function(x) for clf in self.__clfs]
#
#     def decision_function_man(self, x):
#         for sample in x:
#             for clf in self.__clfs:
#                 print(self.evaluate(clf, sample))
#         #print(self.__clfs[0].decision_function(x))
#         #print(self.__clfs[-1].decision_function(x))
#       #  return [clf.decision_function(x) for clf in self.__clfs]
#
#     def kernel(self, params, sv, x):
#         if params['kernel'] == 'linear':
#             return [np.dot(vi, x) for vi in sv]
#         elif params['kernel'] == 'rbf':
#             return [math.exp(-params['gamma'] * np.dot(vi - x, vi - x)) for vi in sv]
#
#     def evaluate(self, clf, x):
#         params = clf.get_params()
#         sv = clf.support_vectors_
#         nv = clf.n_support_
#         a = clf.dual_coef_
#         b = clf._intercept_
#         cs = clf.classes_
#
#         # calculate the kernels
#         k = self.kernel(params, sv, x)
#
#         # define the start and end index for support vectors for each class
#         start = [sum(nv[:i]) for i in range(len(nv))]
#         end = [start[i] + nv[i] for i in range(len(nv))]
#
#         # calculate: sum(a_p * k(x_p, x)) between every 2 classes
#         c = [sum(a[i][p] * k[p] for p in range(start[j], end[j])) +
#              sum(a[j - 1][p] * k[p] for p in range(start[i], end[i]))
#              for i in range(len(nv)) for j in range(i + 1, len(nv))]
#
#         # add the intercept
#         return [sum(x) for x in zip(c, b)]
#

class ConformalSVC(ConformalSVM):

    def __init__(self, nc_measure: NCMeasure, C=1.0, kernel='rbf', degree=3, gamma='auto',
                 coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200,
                 class_weight=None, verbose=False, max_iter=-1, random_state=None):
        super().__init__(nc_measure,
                         SVC(C=C, kernel=kernel, degree=degree, gamma=gamma,
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


