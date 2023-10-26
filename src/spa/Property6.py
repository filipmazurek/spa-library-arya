import abc
import numpy as np
from typing import Any, List, Tuple, Union
import pandas as pd
from spa.properties import GenericProperty, BaseProperty, OutOfDataException

_T_NUMBER = Union[int, float]

class Property6Threshold(BaseProperty, abc.ABC):
    
    """Check if Probability [event occurs within C cycles of previous event] < threshold """
    def conditional(self, values, C, threshold):
        satisfy_trials = 0
        prev = values[0]

        for x in range(1, len(values)):
            if values[x]-prev <= C:
                satisfy_trials+=1
            prev = values[x]

        return (satisfy_trials/len(values)) < threshold

    def __init__(self, C: _T_NUMBER = None, threshold: _T_NUMBER = None):
        """
        :param C: Check if event occurred within C cycles of previous event
        :param threshold: The property's probability threshold to check against.
        """

        if not (isinstance(C, int) or isinstance(C, float) or C is None):
            raise TypeError('C must be an integer or float')
        
        if not (isinstance(threshold, int) or isinstance(threshold, float) or threshold is None):
            raise TypeError('Threshold must be an integer or float')

        # The C cycles and threshold to check against
        self.C = C
        self.threshold = threshold
        self._comparison = self.conditional
        self._high_threshold_outcome = True
    
    @property
    def high_threshold_outcome(self) -> bool:
        """Property result when checking against a high threshold. Depends only on the comparison operator."""
        return self._high_threshold_outcome

class Property6(Property6Threshold, GenericProperty):
    """Property comparison model to compare a single execution property against a threshold.
    Table 1, item 6 in the paper https://doi.org/10.1145/3613424.3623785."""

    @staticmethod
    def start_point_estimate(data: pd.DataFrame, proportion: float) -> float:
        """When using SPA to find a confidence interval, an initial point estimate is needed. This method estimates a
        starting point for the property's true value. In this case, the proportion can be thought of as the inverse of
        quantile.
        :param data: The data to use for the estimate.
        :param proportion: The proportion of the data to use for the estimate.
        :return: The estimated starting point.
        """

        # Get a list of average # of cycles between errors for every individual run
        data_list = []
        for x in pd.unique(pd.Series(data['run'])):

            # Create a list of cycles between each error
            error_data = data.query('run == ' + str(x) + ' and tag == "error"')['value'].tolist()
            delta_values = []
            prev = error_data[0]
            for y in range(1, len(error_data)):
                delta_values.append(error_data[y] - prev)
                prev = error_data[y]

            # Get average of values
            data_list.append(sum(delta_values)/len(delta_values))
        
        return np.quantile(data_list, 1 - proportion)

    def extract_value(self, data: pd.DataFrame) -> Tuple[List[_T_NUMBER], List[_T_NUMBER]]:
        """Extract the value from the input data. Meant to be used in conjunction with :function check_sample_satisfy:
        For this property, return only the leftmost value in the data. Returns the value and the remaining data.
        :param data: The data to extract from.
        :return: The extracted value(s).
        """

        if data is None:
            raise OutOfDataException

        # Get the data from only the top run
        run_number = int(data.iloc[0]['run'])
        run_data = data.query('run == ' + str(run_number))
        if not len(run_data) > 0:
            raise OutOfDataException

        # Get and return the list of cycle numbers for every error
        value = data.query('run == ' + str(run_number) + ' and tag == "error"')['value'].tolist()

        # Remove all the data from the currently used run so we can start on the next run on next data extract
        start_index = data.query('run == ' + str(run_number)).index[0]
        if len(data.query('run == ' + str(run_number+1))) > 0:
            end_index = data.query('run == ' + str(run_number+1)).index[0]
            for x in range(start_index, end_index):
                data = data.drop(x)
        else:
            data = None

        return value, data

    def check_sample_satisfy(self, value: List[_T_NUMBER]) -> bool:
        """Check if the property is satisfied or not satisfied by the given value. Meant to be used with the
        :function extract_value: method.
        In this case, the property is satisfied if the value comparison against the threshold is True.
        :param value: The value(s) to check.
        :return: True if the property is satisfied, False otherwise.
        """
        # First ensure that the property is set
        if not (isinstance(self.C, int) or isinstance(self.C, float)):
            raise TypeError('C must be an integer or float')
        
        if not (isinstance(self.threshold, int) or isinstance(self.threshold, float)):
            raise TypeError('Threshold must be an integer or float')
        
        # Use the comparison operator defined in the constructor to check the value against the property
        return self._comparison(value, self.C, self.threshold)
    
