class NotCalibratedError(Exception):
    """
    Exception that will be thrown when a conformal predictor is not calibrated
    """

    def __init__(self) -> Exception:
        """

        :rtype: Exception
        """
        super(Exception, self).__init__("You must calibrate the Conformal"
                                        " Predictor first")
