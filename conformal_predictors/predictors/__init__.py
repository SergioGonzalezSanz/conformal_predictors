from abc import ABCMeta, abstractmethod
from conformal_predictors.validation import NotCalibratedError
from conformal_predictors.nc_measures import NCMeasure
import numpy as np
from decimal import *


class ConformalPredictor:
    """
    Abstract class that represents the Conformal Predictors and defines their
    basic functionality
    """

    _metaclass_ = ABCMeta

    def __init__(self, nc_measure: NCMeasure, clf) -> None:
        """
        Initialises the conformal predictor

        :param nc_measure: nonconformity measure to be used
        :param clf: classifier
        """
        self._nc_measure = nc_measure  # Non-conformity measure
        self._clf = clf                # Classifier
        self._cal_l = None             # Calibration labels
        self._cal_a = None             # Calibration alphas

    def is_calibrated(self):
        """
        Establish if a conformal predictor is calibrated (the calibration set
        is not null)

        :return: 'True' if the calibration set is populated, ´false´ otherwise
        """
        return not ((self._cal_a is None) or
                    (self._cal_l is None))

    def check_calibrated(self):
        """
        This method raises a NotCalibratedError if the conformal predictor is
        not calibrated
        :return:
        """
        if not self.is_calibrated():
            raise NotCalibratedError()

    @abstractmethod
    def predict_cf(self, x, **kwargs):
        """
        It classifies the samples on X and adds conformal measures

        :param x:
        :return:
        """
        pass

    @abstractmethod
    def calibrate(self, x, y) -> None:
        """
        Method that calibrates the conformal predictor using the calibration
        set in x

        :param x: array-like, shape (n_samples, n_features)
        :return: nothing
        """
        pass

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray, sample_weight: np.array = np.empty(0)) -> None:
        """
        Fits the SVM model according to the given training data

        :param x: {array-like, sparse matrix}, shape (n_samples, n_features).
        Training vectors, where n_samples is the number of samples and
        n_features is the number of features. For kernel="precomputed", the
        expected shape of X is (n_samples, n_samples).
        :param y: array-like, shape (n_samples,). Target values (class labels
        in classification, real number in regression)
        :param sample_weight: array-like, shape (n_samples,). Per-sample
        weights. Rescale C per sample. Higher weights force the classifier to
        put more emphasis on these points.
        :return: nothing
        """
        pass

    def compute_pvalue(self, alphas) -> float:
        # If the model is not calibrated we throw an exception
        self.check_calibrated()
        resolution = 10000
        pvalues = np.array([0.0] * len(alphas))
        for i in range(0, len(alphas)):
            index = np.ix_(self._cal_l == self._clf.classes_[i])
            # print(np.array(self._cal_a)[index])
            calibration = np.round((np.array(self._cal_a)[index] * resolution).astype(int))
            alpha = int(np.round(alphas[i] * resolution))
            pvalues[i] = sum(calibration >= alpha) / (len(calibration) * 1.0)
        # print(alphas)
        # print(pvalues)
        return pvalues
