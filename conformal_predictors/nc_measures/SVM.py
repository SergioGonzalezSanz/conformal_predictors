from conformal_predictors.nc_measures import NCMeasure
import numpy as np
import math


class SVCDistanceNCMeasure(NCMeasure):
    """
    Nonconformity measure based on the distance of the samples to the
    separation hyperplane
    """

    from sklearn.svm import SVC

    # def evaluate(self, clf: SVC, x) -> np.ndarray:
    #     dict = clf.classes_.tolist()        # Classes labels
    #
    #     # dist.shape = (n_samples, n_classes)
    #     # except when no. classes is 2 that the shape is (n_samples,)
    #     print(x)
    #     dist = clf.decision_function(x)     # Distances to hyperplanes
    #     print("Distance")
    #     print(dist)
    #     if len(dict) == 2:
    #         new_dist = np.zeros((x.shape[0], 2))
    #         new_dist[:, 0] = dist
    #         new_dist[:, 1] = dist
    #         dist = np.abs(new_dist)
    #
    #     predicted_l = clf.predict(x)        # Predicted labels
    #     multipliers = np.ones((x.shape[0], len(dict)))
    #
    #     for i in range(0, x.shape[0]):
    #         # The nonconformity measure of the predicted class is equal to
    #         # minus the distance from the sample to the hyperplane
    #         multipliers[i, dict.index(predicted_l[i])] *= -1
    #
    #     return multipliers * dist

    @staticmethod
    def kernel(params, sv, x):
        if params['kernel'] == 'linear':
            return [np.dot(vi, x) for vi in sv]
        elif params['kernel'] == 'rbf':
            return [math.exp(-params['gamma'] * np.dot(vi - x, vi - x))
                    for vi in sv]

    def evaluate(self, clf, x):
        x = x[0, :]
        params = clf.get_params()
        sv = clf.support_vectors_
        nv = clf.n_support_
        a = clf.dual_coef_
        b = clf._intercept_
        cs = clf.classes_
        print(cs)
        print(b)
        print(a)
        print(nv)
        print(sv)
        # calculate the kernels
        k = self.kernel(params, sv, x)
        print(k)

        # define the start and end index for support vectors for each class
        start = [sum(nv[:i]) for i in range(len(nv))]
        end = [start[i] + nv[i] for i in range(len(nv))]

        # calculate: sum(a_p * k(x_p, x)) between every 2 classes
        c = [sum(a[i][p] * k[p] for p in range(start[j], end[j])) +
             sum(a[j - 1][p] * k[p] for p in range(start[i], end[i]))
             for i in range(len(nv)) for j in range(i + 1, len(nv))]

        # add the intercept
        print([sum(x) for x in zip(c, b)])
        return [sum(x) for x in zip(c, b)]
