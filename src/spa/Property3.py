import numpy as np
from typing import Any, List, Union
import pandas as pd
from spa.properties import GenericThreshold, GenericProperty, OutOfDataException

_T_NUMBER = Union[int, float]

class Property3(GenericThreshold, GenericProperty):
    """Property comparison model to compare a single execution property against a upper and lower bound threshold.
    Table 1, item 3 in the paper https://doi.org/10.1145/3613424.3623785."""

    @staticmethod
    def start_point_estimate(data: pd.DataFrame, proportion: float) -> float:
        """When using SPA to find a confidence interval, an initial point estimate is needed. This method estimates a
        starting point for the property's true value. In this case, the proportion can be thought of as the inverse of
        quantile once the data is pre-processed to take the ratio of all data points.
        :param data: The data to use for the estimate.
        :param proportion: The proportion of the data to use for the estimate.
        :return: The estimated starting point.
        """

        # Get a list of % of runtime in mispredict state for every individual run
        data_list = []
        for x in pd.unique(pd.Series(data['run'])):

            # Get start and end cycle value
            start = data.query('run == ' + str(x) + ' and tag == "system start"')['value'].tolist()[0]
            end = data.query('run == ' + str(x) + ' and tag == "system end"')['value'].tolist()[0]

            # Get list of mispredict and recovery cycles
            mispredict_data = data.query('run == ' + str(x) + ' and tag == "mispredict"')['value'].tolist()
            recover_data = data.query('run == ' + str(x) + ' and tag == "recovered"')['value'].tolist()

            # Calculate % of total runtime spend between mispredict and recovery
            delta_values = []
            for y in range(0, len(recover_data)):
                delta_values.append(recover_data[y] - mispredict_data[y])
            data_list.append(sum(delta_values)/(end-start))
        
        return np.quantile(data_list, 1 - proportion)

    def extract_value(self, data: pd.DataFrame):
        """Extract the value from the input data. Meant to be used in conjunction with :function check_sample_satisfy:
        For this property, return only the leftmost value in both data lists. Returns the value and the remaining data.
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

        # Calculate % of runtime in mispredict state for the run
        start = data.query('run == ' + str(run_number) + ' and tag == "system start"')['value'].tolist()[0]
        end = data.query('run == ' + str(run_number) + ' and tag == "system end"')['value'].tolist()[0]
        mispredict_data = run_data.query('run == ' + str(run_number) + ' and tag == "mispredict"')['value'].tolist()
        recover_data = run_data.query('run == ' + str(run_number) + ' and tag == "recovered"')['value'].tolist()
        delta_values = []
        for y in range(0, len(recover_data)):
            delta_values.append(recover_data[y] - mispredict_data[y])
        value = sum(delta_values)/(end-start)

        # Remove all the data from the currently used run so we can start on the next run on next data extract
        start_index = data.query('run == ' + str(run_number)).index[0]
        if len(data.query('run == ' + str(run_number+1))) > 0:
            end_index = data.query('run == ' + str(run_number+1)).index[0]
            for x in range(start_index, end_index):
                data = data.drop(x)
        else:
            data = None

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
        
        # Use the comparison operator defined in the constructor to check the value against the threshold
        return self._comparison(value, self.threshold)