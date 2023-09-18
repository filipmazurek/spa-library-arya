from math import ceil
from sympy import symbols, solve
from typing import Union

_T_NUMBER = Union[int, float]


def min_num_samples(prob_threshold: float, confidence: float) -> int:
    """Find the minimum number of samples to reach a conclusion given the proportion and confidence of the SMC run.
    Based on the number for minimum number of false results to reach a conclusion, and the same for true results.
    This equation is discussed in Section 4.3 of https://doi.org/10.1145/3613424.3623785.
    :param prob_threshold: The proportion of samples that satisfy the property
    :param confidence: The confidence level of the SMC run
    :return: The minimum number of samples to reach a conclusion
    """

    n = symbols('n')
    expr_false = - confidence + 1 - (1 - prob_threshold) ** n
    expr_true = - confidence + 1 - prob_threshold ** n

    sol_false = solve(expr_false)
    num_false = ceil(sol_false[0])

    sol_true = solve(expr_true)
    num_true = ceil(sol_true[0])

    return max(num_false, num_true)


def round_to(val: _T_NUMBER, precision: _T_NUMBER) -> float:
    """Rounding function to the nearest multiple of the input precision
    :param val: The value to round
    :param precision: The precision to round to
    :return: The rounded value
    """

    return round(val / precision) * precision


def sort_dict_by_key(dictionary: dict) -> dict:
    """Sort a dictionary by its keys
    :param dictionary: The dictionary to sort
    :return: The sorted dictionary
    """

    return {k: v for k, v in sorted(dictionary.items(), key=lambda item: item[0])}
