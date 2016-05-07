from abc import ABCMeta, abstractmethod
from conformal_predictors.validation import NotCalibratedError
from conformal_predictors.nc_measures.SVM import SVCDistanceNCMeasure
from conformal_predictors.nc_measures import NCMeasure
from sklearn.svm import SVC
import numpy as np


class ConformalPredictor:
    """
    Abstract class that represents the Conformal Predictors and defines their
    basic functionality
    """

    _metaclass_ = ABCMeta

    def __init__(self, clf, nc_measure: NCMeasure) -> None:
        """
        Initialises the conformal predictor

        :param nc_measure: Nonconformity measure to be used
        """
        self._clf = clf                # Classifier
        self._cal_l = None             # Calibration labels
        self._cal_a = None             # Calibration alphas
        self._nc_measure = nc_measure  # Non-conformity measure

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
        :return: Nothing
        """
        pass

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray,
            sample_weight: np.array = np.empty(0)) -> None:
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
        :return: None
        """
        pass

    def compute_pvalue(self, alpha) -> float:
        self.check_calibrated()
        return sum(self._cal_a > alpha) / (len(self._cal_a) * 1.0)


class ConformalSVCPredictor(ConformalPredictor):

    def __init__(self, nc_measure: NCMeasure):
        super().__init__(SVC(decision_function_shape='ovr'), nc_measure)

    def fit(self, x: np.ndarray, y: np.ndarray,
            sample_weight: np.array = np.empty(0)) -> None:
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
        p_values = np.zeros(alphas.shape)
        predicted_l = np.zeros((x.shape[0]))
        confidence = np.zeros((x.shape[0]))
        credibility = np.zeros((x.shape[0]))
        for i in range(0, alphas.shape[0]):
            for j in range(0, alphas.shape[1]):
                p_values[i, j] = self.compute_pvalue(alphas[i, j])
            conformal = np.sort(p_values[i, :])
            predicted_l[i] = dict[np.argmax(p_values[i, :])]
            credibility[i] = conformal[-1]
            confidence[i] = 1.0 - conformal[-2]

        return [predicted_l, credibility, confidence]


dict = [3,1,2]
x = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
z = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
y = np.array([3,98, 2, 2])
nc = SVCDistanceNCMeasure()
cp = ConformalSVCPredictor(nc)
cp.fit(x,y)

cp.calibrate(z,[3,98,98,2])
print(cp.predict_cf(x))



