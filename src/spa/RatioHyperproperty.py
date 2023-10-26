import numpy as np
from typing import Any, List, Union
import pandas as pd
from spa.properties import GenericThreshold, OutOfDataException

_T_NUMBER = Union[int, float]

class RatioHyperproperty(GenericThreshold):
    """Comparison model to compare the ratios of two execution properties. Notably, this includes speedup.
    Technically, the data may be pre-processed to take the ratio of all data points and then the property
    :class ThresholdProperty can be used. However, this class is provided as an example of a hyperproperty.
    """

    NUM_SOURCES = 2
    """The expected number of data sources for the property"""

    @staticmethod
    def start_point_estimate(data: pd.DataFrame, proportion: float) -> float:
        """When using SPA to find a confidence interval, an initial point estimate is needed. This method estimates a
        starting point for the property's true value. In this case, the proportion can be thought of as the inverse of
        quantile once the data is pre-processed to take the ratio of all data points.
        :param data: The data to use for the estimate.
        :param proportion: The proportion of the data to use for the estimate.
        :return: The estimated starting point.
        """

        # Take all the ratios of the data
        data1 = data.query('System == 1')['value'].tolist()
        data2 = data.query('System == 2')['value'].tolist()

        ratio_data = []
        for x in range(0, min(len(data1), len(data2))):
            ratio_data.append(data2[x] / data1[x])
        return np.quantile(ratio_data, 1 - proportion)

    def verify_data(self, data: pd.DataFrame) -> None:
        """Verify that the input data is valid for the property. Raises an exception to provide detailed error info.
        In this case, expecting a pandas dataframe with the correct headers and float or integer data
        :param data: The data to verify.
        :raises TypeError: If the data is not a tuple or list of integers or floats of length at least 1.
        """

        correct_type = isinstance(data, pd.DataFrame)
        correct_headers = set(data.columns) == set(['System', 'run', 'value', 'tag', "Unnamed: 4"])
        correct_length = len(pd.unique(data['System'])) == self.NUM_SOURCES
        correct_data_type = all(isinstance(x, int) or isinstance(x, float) for x in data['value'].tolist())
        correct_sub_length = all(len(data.query('System == ' + str(x))) >= 1 for x in pd.unique(pd.Series(data['System'])))

        if not (correct_type and correct_headers and correct_length and correct_data_type and correct_sub_length):
            raise TypeError(
                'Data must be a list or tuple of lists or tuples of integers or floats of length at least 1')

    def extract_value(self, data: pd.DataFrame):
        """Extract the value from the input data. Meant to be used in conjunction with :function check_sample_satisfy:
        For this property, return only first value of both systems from the dataframe. Returns the value and the remaining data.
        :param data: The data to extract from.
        :return: The extracted value(s).
        """
        # Check if each data source has at least one value
        has_data = all(len(data.query('System == ' + str(x))) > 1 for x in pd.unique(pd.Series(data['System'])))
        if not has_data:
            raise OutOfDataException
        
        # Extract the first value from each unique system data source
        value = []
        for x in pd.unique(pd.Series(data['System'])):
            target_row = data.query('System == ' + str(x)).iloc[0]
            value.append(target_row['value'])

            # Clear the data that was used
            data = data.drop(data.query("System == "+str(target_row['System']) + " and run == " + str(target_row['run'])).index[0])
        return value, data

    def check_sample_satisfy(self, value):
        """Check if the property is satisfied or not satisfied by the given value. Meant to be used with the
        :function extract_value: method.
        In this case, the property is satisfied if the value comparison against the threshold is True.
        :param value: The value(s) to check.
        :return: True if the property is satisfied, False otherwise.
        """

        # First ensure that the property is set
        if not (isinstance(self.threshold, int) or isinstance(self.threshold, float)):
            raise TypeError('Threshold must be an integer or float')
        
        # Ratio is taken of the first data source over the second
        ratio = value[0] / value[1]
        return self._comparison(ratio, self.threshold)
