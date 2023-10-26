import abc
import numpy as np
from typing import Any, List, Tuple, Union
import pandas as pd
from spa.properties import GenericProperty, BaseProperty, OutOfDataException

_T_NUMBER = Union[int, float]

class Property2Threshold(BaseProperty, abc.ABC):

    """Check if b < a < c"""
    def between(self, a, b, c):
        return a > b and a < c

    def __init__(self, threshold1: _T_NUMBER = None, threshold2: _T_NUMBER = None):
        """
        :param threshold1: The property's lower bound threshold to check against
        :param threshold2: The property's upper bound threshold to check against
        """
        if not (isinstance(threshold1, int) or isinstance(threshold1, float) or threshold1 is None):
            raise TypeError('Threshold 1 must be an integer or float')
        
        if not (isinstance(threshold2, int) or isinstance(threshold2, float) or threshold2 is None):
            raise TypeError('Threshold 2 must be an integer or float')

        # The lower and upper bound thresholds to check against
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        self._comparison = self.between
        self._high_threshold_outcome = False
    
    @property
    def high_threshold_outcome(self) -> bool:
        """Property result when checking against a high threshold. Depends only on the comparison operator."""
        return self._high_threshold_outcome

class Property2(Property2Threshold, GenericProperty):
    """Property comparison model to compare a single execution property against a upper and lower bound threshold.
    Table 1, item 2 in the paper https://doi.org/10.1145/3613424.3623785."""

    @staticmethod
    def start_point_estimate(data: pd.DataFrame, proportion: float) -> float:
        """When using SPA to find a confidence interval, an initial point estimate is needed. This method estimates a
        starting point for the property's true value. In this case, the proportion can be thought of as the inverse of
        quantile.
        :param data: The data to use for the estimate.
        :param proportion: The proportion of the data to use for the estimate.
        :return: The estimated starting point.
        """

        data = data.query("run == 1 and tag == 'cycles'")["value"].tolist()

        return np.quantile(data, 1 - proportion)

    def extract_value(self, data: pd.DataFrame) -> Tuple[_T_NUMBER, List[_T_NUMBER]]:
        """Extract the value from the input data. Meant to be used in conjunction with :function check_sample_satisfy:
        For this property, return only first value of both systems from the dataframe. Returns the value and the remaining data.
        :param data: The data to extract from.
        :return: The extracted value(s).
        """

        if data.shape[0] < 1:
            raise OutOfDataException
        
        # Read the first value and clear it from returned data set
        parsed_data = data.query("run == 1 and tag == 'cycles'")
        value = parsed_data.iloc[0]
        return value, parsed_data.iloc[1:]

    def check_sample_satisfy(self, value: _T_NUMBER) -> bool:
        """Check if the property is satisfied or not satisfied by the given value. Meant to be used with the
        :function extract_value: method.
        In this case, the property is satisfied if the value comparison against the threshold is True.
        :param value: The value(s) to check.
        :return: True if the property is satisfied, False otherwise.
        """

        # First ensure that the property is set
        if not (isinstance(self.threshold1, int) or isinstance(self.threshold1, float)):
            raise TypeError('Threshold 1 must be an integer or float')
        if not (isinstance(self.threshold2, int) or isinstance(self.threshold2, float)):
            raise TypeError('Threshold 2 must be an integer or float')
        
        # Use the comparison operator defined in the constructor to check the value against the property
        return self._comparison(value["value"], self.threshold1, self.threshold2)
    
