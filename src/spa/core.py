from math import ceil, log10
import operator
from scipy.stats import beta
from typing import Dict

from spa.dataclasses import ConfidenceInterval, SPAResult, SMCResult, create_smc_result
from spa.properties import OutOfDataException, BaseProperty
from spa.util import round_to, sort_dict_by_key


class SPAException(Exception):
    """Exception for all SPA-related errors"""
    pass


def _smc_iv(data, property_: BaseProperty, prob_threshold: float, desired_conf: float, continuous: bool) -> None:
    """SMC Input Validation"""
    # Check that the data is in the correct form
    property_.verify_data(data)
    # Check that all parameters are valid
    if not (0 <= prob_threshold <= 1):
        raise ValueError('Probability threshold must be between 0 and 1')
    if not (0 <= desired_conf <= 1):
        raise ValueError('Desired confidence must be between 0 and 1')
    if not isinstance(continuous, bool):
        raise ValueError('Continuous must be a boolean')


def smc(data, property_: BaseProperty, prob_threshold: float, confidence: float, *,
        continuous: bool = False) -> SMCResult:
    """The SMC algorithm as described in https://doi.org/10.1145/3613424.3623785.
    Given data and the property to check, this function will check if the specified property holds with the given
    probability threshold and confidence level. The data given must be in the form specified by the chosen property.
    :param data: The data to check. Must be in the form specified by the chosen property.
    :param property_: The property to check. An instantiated property class.
    :param prob_threshold: The proportion of samples that satisfy the property.
    :param confidence: The desired confidence level to reach.
    :param continuous: Whether to run the algorithm even after the desired confidence is reached. In regular SMC, the
    algorithm terminates once the desired confidence is reached. In order to find the confidence interval, this must be
    set to True so that all SMC runs use the same data.
    :return: Result packaged in a SMCResult object.
    """
    # Validate the input
    _smc_iv(data, property_, prob_threshold, confidence, continuous)

    result = None  # Keep track of the intermediate algorithm result
    num_trials = 0  # Total number of trials run
    satisfied_trials = 0  # Number of trials satisfying condition
    conf_cp = 0  # Confidence level at current iteration (Clopper-Pearson)
    conf_cp_list = []  # List of confidence level at each iteration, may be tracked for plotting
    # List of if the algorithm is leaning towards true or false at each iteration. Necessary as the confidence level
    #   is for the corresponding result here.
    lean_list = []
    satisfied_list = []  # List of if the sample satisfies the property at each iteration

    while (conf_cp < confidence) or continuous:  # Loop until confidence level is reached
        sample_satisfy = None

        # Loop until the property has enough data to check. May be necessary in cases where the property checks the
        #   number of cycles between values, for example
        while sample_satisfy is None:
            # Get value and update new data list
            try:
                value, data = property_.extract_value(data)
            except OutOfDataException:
                # Once the data is exhausted, return the result. Needed in cases where no conclusion is reached or
                #   continuous is set to True
                if conf_cp < confidence:
                    result = None
                else:
                    result = (satisfied_trials / num_trials) > prob_threshold
                # Wrap into the SMCResult class
                return create_smc_result(result, conf_cp, num_trials, satisfied_trials, conf_cp_list, lean_list)

            sample_satisfy = property_.check_sample_satisfy(value)

        if sample_satisfy:
            satisfied_trials += 1

        satisfied_list.append(sample_satisfy)
        num_trials += 1

        # Compute the new significance level
        if satisfied_trials / num_trials < prob_threshold:
            a, b = (0, prob_threshold)
        else:
            a, b = (prob_threshold, 1)

        # Calculate the new confidence
        if satisfied_trials == 0:
            conf_cp = pow(1 - a, num_trials) - pow(1 - b, num_trials)
        elif satisfied_trials == num_trials:
            conf_cp = pow(b, num_trials) - pow(a, num_trials)
        else:
            dist_a = beta.cdf(b, satisfied_trials + 1, num_trials - satisfied_trials)
            dist_b = beta.cdf(a, satisfied_trials, num_trials - satisfied_trials + 1)
            conf_cp = dist_a - dist_b

        conf_cp_list.append(conf_cp)

        # Check current result
        result = (satisfied_trials / num_trials) > prob_threshold
        lean_list.append(result)

    return create_smc_result(result, conf_cp, num_trials, satisfied_trials, conf_cp_list, lean_list)


def _linear_search(data, property_: BaseProperty, prob_threshold: float, confidence: float,
                   iteration_limit: int, granularity: float, search_start_point: float,
                   direction_operator, goal_bool: bool) -> (Dict[float, SMCResult]):
    # Linear search in one direction
    if not ((direction_operator is operator.add) or (direction_operator is operator.sub)):
        raise TypeError('direction operator must be either add or sub')

    smc_res_detail = {}
    param = search_start_point
    iteration = 0

    # Set the initial parameter
    property_.threshold = param

    while iteration < iteration_limit:
        # Run SMC to find if the result is True, False, or inconclusive
        smc_result = smc(data, property_, prob_threshold, confidence, continuous=True)

        smc_res_detail[param] = smc_result

        # Stop when the goal state is hit
        if smc_result.result is goal_bool:
            return smc_res_detail

        # Update property threshold for the next iteration
        param = round_to(direction_operator(param, granularity), granularity)
        property_.threshold = param
        iteration += 1

    raise SPAException('Reached search iteration limit without finding the goal SMC result. '
                       'Possible debug steps: '
                       '1. Check that the property has enough data to converge (run the :function smc: method). '
                       '2. Consider raising the iteration limit. '
                       '3. Consider providing a start point.')


def _spa_iv(data, property_: BaseProperty, prob_threshold: float, confidence: float, iteration_limit: int,
            granularity: float, search_start_point: float) -> None:
    if not(granularity is None):
        if granularity <= 0:
            raise ValueError('granularity must be > 0')


def spa(data, property_: BaseProperty, prob_threshold: float, confidence: float = 0.9, *, iteration_limit: int = 1000,
        granularity: float = None, search_start_point: float = None) -> SPAResult:
    """The SMC-based technique of finding confidence intervals, as described in https://doi.org/10.1145/3613424.3623785.
    Using the SMC hypothesis-testing algorithm, this method tests multiple threshold values for a property and creates
    a confidence interval based on the results.
    :param data: The data to be tested
    :param property_: SMC property to perform hypothesis tests
    :param prob_threshold: The probability threshold for the property to be satisfied
    :param confidence: The confidence level desired for the confidence interval
    :param iteration_limit: The maximum number of search iterations to run
    :param granularity: The granularity of the search
    :param search_start_point: The starting point of the search
    :return: A SPAResult object containing the confidence interval and details of each SMC result for each threshold
    """
    # Input validation
    _spa_iv(data, property_, prob_threshold, confidence, iteration_limit, granularity, search_start_point)

    if search_start_point is None:
        search_start_point = property_.start_point_estimate(data, prob_threshold)

    # Calculate granularity that is the nearest order of magnitude to 0.1% of the mean value
    if granularity is None:
        granularity = 10 ** ceil(log10(search_start_point / 1000))

    # Update the starting point to the nearest multiple of the granularity
    search_start_point = round_to(search_start_point, granularity)

    # Find the goal values based on the high_threshold_outcome
    goal_bool_incr = property_.high_threshold_outcome
    goal_bool_decr = not property_.high_threshold_outcome

    # Find the goal SMC result in each direction
    # TODO: Create a better search method
    smc_dict_incr = _linear_search(data=data, property_=property_, prob_threshold=prob_threshold,
                                   confidence=confidence, iteration_limit=iteration_limit,
                                   granularity=granularity, search_start_point=search_start_point,
                                   direction_operator=operator.add, goal_bool=goal_bool_incr)
    search_start_point = round_to(search_start_point - granularity, granularity)
    smc_dict_decr = _linear_search(data=data, property_=property_, prob_threshold=prob_threshold,
                                   confidence=confidence, iteration_limit=iteration_limit,
                                   granularity=granularity, search_start_point=search_start_point,
                                   direction_operator=operator.sub, goal_bool=goal_bool_decr)

    # Combine the results from both directions
    smc_dict = sort_dict_by_key({**smc_dict_decr, **smc_dict_incr})

    # Find the confidence interval
    smc_dict_items = smc_dict.items()
    if goal_bool_incr:
        low = max((key for key, value in smc_dict_items if value.result is False))
        high = min((key for key, value in smc_dict_items if value.result is True))
    else:
        low = max((key for key, value in smc_dict_items if value.result is True))
        high = min((key for key, value in smc_dict_items if value.result is False))

    return SPAResult(confidence_interval=ConfidenceInterval(low=low, high=high), result_detail=smc_dict)
