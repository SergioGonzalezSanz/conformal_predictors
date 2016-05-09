from conformal_predictors.nc_measures import NCMeasure
import numpy as np


class SVCDistanceNCMeasure(NCMeasure):
    """
    Nonconformity measure based on the distance of the samples to the
    separation hyperplane
    """

    from sklearn.svm import SVC

    def evaluate(self, clf: SVC, x) -> np.ndarray:
        dict = clf.classes_.tolist()        # Classes labels

        # dist.shape = (n_samples, n_classes)
        # except when no. classes is 2 that the shape is (n_samples,)
        print(x)
        dist = clf.decision_function(x)     # Distances to hyperplanes
        print("Distance")
        print(dist)
        if len(dict) == 2:
            new_dist = np.zeros((x.shape[0], 2))
            new_dist[:, 0] = dist
            new_dist[:, 1] = dist
            dist = np.abs(new_dist)

        predicted_l = clf.predict(x)        # Predicted labels
        multipliers = np.ones((x.shape[0], len(dict)))

        for i in range(0, x.shape[0]):
            # The nonconformity measure of the predicted class is equal to
            # minus the distance from the sample to the hyperplane
            multipliers[i, dict.index(predicted_l[i])] *= -1

        return multipliers * dist
