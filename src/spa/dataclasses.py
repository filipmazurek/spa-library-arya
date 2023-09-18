from collections import namedtuple
from dataclasses import dataclass
from typing import Dict, List

ConfidenceInterval = namedtuple('ConfidenceInterval', ['low', 'high'])
SMCInternalDetail = namedtuple('ResultDetail', ['num_trials', 'satisfied_trials', 'conf_cp_list', 'lean_list'])


@dataclass
class SMCResult:
    """Result of the SMC algorithm"""
    result: bool
    conf: float
    result_detail: SMCInternalDetail


@dataclass
class SPAResult:
    """Result of the SPA algorithm"""
    confidence_interval: ConfidenceInterval
    result_detail: Dict[float, SMCResult]


def create_smc_result(result: bool, conf: float, num_trials: int, satisfied_trials: int, conf_cp_list: List[float],
                      lean_list: List[bool]) -> SMCResult:
    return SMCResult(result, conf, SMCInternalDetail(num_trials, satisfied_trials, conf_cp_list, lean_list))
