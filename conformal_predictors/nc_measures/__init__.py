class NCMeasure:
    """
    This abstract class defines the behaviour of a nonconformity measure
    """

    from abc import ABCMeta, abstractmethod

    __metaclass__ = ABCMeta

    def __init__(self):
        return None

    @abstractmethod
    def evaluate(self, clf, x) -> float:
        """
        Method that computes the evaluation of the nonconformity
        measure.

        :param clf: classifier
        :param x: values to be evaluated, shape (n_samples, n_features)
        :return: nonconformity measures of x, shape (n_samples, n_classes)
        """
        pass
