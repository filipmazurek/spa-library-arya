import abc
import operator
import numpy as np
from typing import Any, List, Tuple, Union

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

    def verify_data(self, data: List[_T_NUMBER]):
        """Verify that the input data is valid for the property. Raises an exception to provide detailed error info.
        In this case, expecting a single-dimensional list of numbers.
        :param data: The data to verify.
        :raises TypeError: If the data is not a tuple or list of integers or floats of length at least 1.
        """

        correct_type = isinstance(data, tuple) or isinstance(data, list)
        correct_length = len(data) >= 1
        correct_data_type = all(isinstance(x, int) or isinstance(x, float) for x in data)
        if not (correct_type and correct_length and correct_data_type):
            raise TypeError('Data must be a tuple or list of integers or floats of length at least 1')


class GenericThreshold(BaseProperty, abc.ABC):
    """Generic (hyper)property class which requires the input of some threshold to check against."""

    def __init__(self, threshold: _T_NUMBER = None, op: str = '>'):
        """
        :param threshold: The property's current threshold to check against. May be changed later.
        :param op: The comparison operator to use for the threshold comparison. '>' or '<'.
        """
        # TODO: change to isinstance(threshold, _T_NUMBER) when we can move the Python version to 3.10
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


class ThresholdProperty(GenericThreshold, GenericProperty):
    """Property comparison model to compare a single execution property against a threshold.
    Table 1, item 1 in the paper https://doi.org/10.1145/3613424.3623785."""

    @staticmethod
    def start_point_estimate(data: List[_T_NUMBER], proportion: float) -> float:
        """When using SPA to find a confidence interval, an initial point estimate is needed. This method estimates a
        starting point for the property's true value. In this case, the proportion can be thought of as the inverse of
        quantile.
        :param data: The data to use for the estimate.
        :param proportion: The proportion of the data to use for the estimate.
        :return: The estimated starting point.
        """
        return np.quantile(data, 1 - proportion)

    def extract_value(self, data: List[_T_NUMBER]) -> Tuple[_T_NUMBER, List[_T_NUMBER]]:
        """Extract the value from the input data. Meant to be used in conjunction with :function check_sample_satisfy:
        For this property, return only the leftmost value in the data. Returns the value and the remaining data.
        :param data: The data to extract from.
        :return: The extracted value(s).
        """

        if len(data) < 1:
            raise OutOfDataException
        # Read data from left to right
        value = data[0]
        return value, data[1:]

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
        return self._comparison(value, self.threshold)


class RatioHyperproperty(GenericThreshold):
    """Comparison model to compare the ratios of two execution properties. Notably, this includes speedup.
    Technically, the data may be pre-processed to take the ratio of all data points and then the property
    :class ThresholdProperty can be used. However, this class is provided as an example of a hyperproperty.
    """

    NUM_SOURCES = 2
    """The expected number of data sources for the property"""

    @staticmethod
    def start_point_estimate(data: List[List[_T_NUMBER]], proportion: float) -> float:
        """When using SPA to find a confidence interval, an initial point estimate is needed. This method estimates a
        starting point for the property's true value. In this case, the proportion can be thought of as the inverse of
        quantile once the data is pre-processed to take the ratio of all data points.
        :param data: The data to use for the estimate.
        :param proportion: The proportion of the data to use for the estimate.
        :return: The estimated starting point.
        """
        # First take all the ratios of the data
        ratio_data = [x[1] / x[0] for x in zip(*data)]
        return np.quantile(ratio_data, 1 - proportion)

    def verify_data(self, data: List[List[_T_NUMBER]]) -> None:
        """Verify that the input data is valid for the property. Raises an exception to provide detailed error info.
        In this case, expecting two lists, each of which contains at least one integer or float.
        :param data: The data to verify.
        :raises TypeError: If the data is not a tuple or list of integers or floats of length at least 1.
        """

        correct_type = isinstance(data, list) or isinstance(data, list)
        correct_length = len(data) == self.NUM_SOURCES
        correct_data_type = all(isinstance(x, tuple) or isinstance(x, list) for x in data)

        correct_sub_length = all(len(x) >= 1 for x in data)
        correct_sub_type = all(isinstance(y, int) or isinstance(y, float) for x in data for y in x)

        if not (correct_type and correct_length and correct_data_type and correct_sub_length and correct_sub_type):
            raise TypeError(
                'Data must be a list or tuple of lists or tuples of integers or floats of length at least 1')

    def extract_value(self, data: List[List[_T_NUMBER]]):
        """Extract the value from the input data. Meant to be used in conjunction with :function check_sample_satisfy:
        For this property, return only the leftmost value in both data lists. Returns the value and the remaining data.
        :param data: The data to extract from.
        :return: The extracted value(s).
        """
        # Check if each data source has at least one value
        has_data = all(len(d) > 0 for d in data)
        if not has_data:
            raise OutOfDataException
        # Extract the leftmost value from each data source
        value = [d[0] for d in data]
        data = [d[1:] for d in data]
        return value, data

    def check_sample_satisfy(self, value):
        """Check if the property is satisfied or not satisfied by the given value. Meant to be used with the
        :function extract_value: method.
        In this case, the property is satisfied if the value comparison against the threshold is True.
        :param value: The value(s) to check.
        :return: True if the property is satisfied, False otherwise.
        """
        if not (isinstance(self.threshold, int) or isinstance(self.threshold, float)):
            raise TypeError('Threshold must be an integer or float')
        # Ratio is taken of the first data source over the second
        ratio = value[0] / value[1]
        return self._comparison(ratio, self.threshold)
