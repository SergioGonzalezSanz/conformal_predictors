from abc import ABCMeta, abstractmethod
from numpy import array


class NonConformityMeasure:
    """
    This abstract class defines the behaviour of a nonconformity measure
    """
    __metaclass__ = ABCMeta

    def __init__(self):
        return None

    @abstractmethod
    def evaluate(self, a: array, b) -> float:
        """
        Method that computes the evaluation of the nonconformity
        measure.

        :param a: array of values
        :param b: value
        :type arg1: type description
        :type arg1: type description
        :return: return description
        :rtype: the return type description
        :Example:

        evaluate([1,2,3],3)

        """
        pass


class DistanceMeasure(NonConformityMeasure):
    """
    Example of a non conformity measure
    """
    def evaluate(self, a: array, b) -> float:
        return 0


ab = DistanceMeasure()
print(ab.evaluate(3, 3))

