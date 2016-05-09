class Utils:
    """
    This class contains a set of tools to compute conformal predictors
    """

    from numpy import array

    @staticmethod
    def compute_pvalue(bag: array, sample: float) -> float:
        print("hello")
        """
        Function that computes the p-value of a sample given a bag of values
        :param bag: array of nonconformity values
        :param sample: noncormity value of the new sample
        :return: p-value of the new sample
        :example:

        compute_pvalue(numpy.array([0.8, 1.4, 0.5]), 0.3)
        """

        return (sum(bag >= sample) + 1.0) / (len(bag) + 1.0)
