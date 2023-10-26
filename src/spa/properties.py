import abc
import operator
import numpy as np
from typing import Any, List, Tuple, Union
import pandas as pd

_T_NUMBER = Union[int, float]


class OutOfDataException(Exception):
    """Exception raised when the data source is out of data."""
    pass


class BaseProperty(abc.ABC):
    """The most generic property class"""

    NUM_SOURCES = None
    """The expected number of data sources for the property."""

    @property
    def high_threshold_outcome(self) -> bool:
        """Property result when checking against a high threshold. Depends only on the comparison operator.
        """
        raise NotImplementedError

    @staticmethod
    def start_point_estimate(data, proportion: float) -> float:
        """When using SPA to find a confidence interval, an initial point estimate is needed. This method estimates a
        starting point for the property's true value.`
        :param data: The data to use for the estimate.
        :param proportion: The proportion of the data to use for the estimate.
        :return: The estimated starting point.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def verify_data(self, data) -> None:
        """Verify that the input data is valid for the property.
        :param data: The data to verify.

        """
        pass

    @abc.abstractmethod
    def extract_value(self, data) -> Any:
        """Extract the value(s) from the input data. This method ensures that only the data needed for a single
        satisfaction check is extracted. Note this is particularly important for more complex property checks. Meant to
        be used in conjunction with :function check_sample_satisfy:.
        :param data: The data to extract from.
        :return: The extracted value(s).
        """
        pass

    @abc.abstractmethod
    def check_sample_satisfy(self, value: Any) -> bool:
        """Check if the property is satisfied or not satisfied by the given value(s). Meant to be used with the
        :function extract_value: method.
        :param value: The value(s) to check.
        :return: True if the property is satisfied, False otherwise.
        """
        pass


class GenericProperty(BaseProperty, abc.ABC):
    """Generic property class. Works with a single data source."""

    NUM_SOURCES = 1
    """The expected number of data sources for the property"""

    def verify_data(self, data: pd.DataFrame):
        """Verify that the input data is valid for the property. Raises an exception to provide detailed error info.
        In this case, expecting a single-dimensional list of numbers.
        :param data: The data to verify.
        :raises TypeError: If the data is not a tuple or list of integers or floats of length at least 1.
        """
        correct_type = isinstance(data, pd.DataFrame)
        correct_headers = set(data.columns) == set(['System', 'run', 'value', 'tag', "Unnamed: 4"])
        correct_length = len(data) >= 1 or data.shape[0] > 1
        correct_data_type = all(isinstance(x, int) or isinstance(x, float) for x in data['value'].tolist())
        
        if not (correct_type and correct_headers and correct_length and correct_data_type):
            raise TypeError('Data must be a tuple or list of integers or floats of length at least 1')

class GenericThreshold(BaseProperty, abc.ABC):
    """Generic (hyper)property class which requires the input of some threshold to check against."""

    def __init__(self, threshold: _T_NUMBER = None, op: str = '>'):
        """
        :param threshold: The property's current threshold to check against. May be changed later.
        :param op: The comparison operator to use for the threshold comparison. '>' or '<'.
        """
        if not (isinstance(threshold, int) or isinstance(threshold, float) or threshold is None):
            raise TypeError('Threshold must be an integer or float')

        # The threshold is the value to check against
        self.threshold = threshold

        if op == '>':
            # comparison is an operation to decide how to check against the threshold
            self._comparison = operator.gt
            # high_threshold_outcome is the result of the property when checking against a high threshold
            self._high_threshold_outcome = False
        elif op == '<':
            self._comparison = operator.lt
            self._high_threshold_outcome = True
        else:
            raise RuntimeError('Given operation for the property must be either `>` or `<`')

    @property
    def high_threshold_outcome(self) -> bool:
        """Property result when checking against a high threshold. Depends only on the comparison operator."""
        return self._high_threshold_outcome