from conformal_predictors.nc_measures import NCMeasure
import numpy as np

class SVCDistanceNCMeasure(NCMeasure):
    """
    Nonconformity measure based on the distance of the samples to the
    separation hyperplane
    """

    from sklearn.svm import SVC

    def evaluate(self, clf: SVC, x) -> np.ndarray:
        # dist.shape = (n_samples, n_classes)
        dist = clf.decision_function(x)     # Distances to hyperplanes
        dict = clf.classes_.tolist()        # Classes labels
        predicted_l = clf.predict(x)        # Predicted labels
        multipliers = np.ones((x.shape[0], len(dict)))

        for i in range(0, x.shape[0]):
            # The nonconformity measure of the predicted class is equal to
            # minus the distance from the sample to the hyperplane
            multipliers[i, dict.index(predicted_l[i])] *= -1

        return np.array(multipliers * dist, np.float32)
