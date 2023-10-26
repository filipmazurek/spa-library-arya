import numpy as np
from typing import Any, List, Tuple, Union
import pandas as pd
from spa.properties import GenericProperty, GenericThreshold, OutOfDataException

_T_NUMBER = Union[int, float]

class ThresholdProperty(GenericThreshold, GenericProperty):
    """Property comparison model to compare a single execution property against a threshold.
    Table 1, item 1 in the paper https://doi.org/10.1145/3613424.3623785."""

    @staticmethod
    def start_point_estimate(data: pd.DataFrame, proportion: float) -> float:
        """When using SPA to find a confidence interval, an initial point estimate is needed. This method estimates a
        starting point for the property's true value. In this case, the proportion can be thought of as the inverse of
        quantile.
        :param data: The data to use for the estimate.
        :param proportion: The proportion of the data to use for the estimate.
        :return: The estimated starting point.
        """

        data = data.query("System == 1 and tag == 'cycles'")["value"].tolist()
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
        parsed_data = data.query("System == 1 and tag == 'cycles'")
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
        if not (isinstance(self.threshold, int) or isinstance(self.threshold, float)):
            raise TypeError('Threshold must be an integer or float')
        
        # Use the comparison operator defined in the constructor to check the value against the threshold
        return self._comparison(value["value"], self.threshold)