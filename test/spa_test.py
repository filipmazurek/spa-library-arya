import csv
from pathlib import Path
import pytest
from typing import List

from spa.core import SPAException, spa
from spa.properties import ThresholdProperty, RatioHyperproperty

slow_data_file = str(Path(__file__).parent) + '/data/ferret-l2_512kB-simticks.csv'
fast_data_file = str(Path(__file__).parent) + '/data/ferret-l2_3MB-simticks.csv'


def read_data(directory: str) -> List[int]:
    """Reads data from a csv file and returns it as a list of ints"""
    with open(directory, 'r') as file:
        reader = csv.reader(file, dialect='excel')
        string_vals = reader.__next__()

    data = [int(s) for s in string_vals]

    return data


def test_basic_threshold():
    # Find data to analyze
    slow_data = read_data(slow_data_file)

    # Should complete successfully
    spa(slow_data, ThresholdProperty(), prob_threshold=0.9, confidence=0.9)


def test_basic_ratio():
    # Get data
    slow_data = read_data(slow_data_file)
    fast_data = read_data(fast_data_file)

    # Prepare RatioHyperproperty data
    data = [slow_data, fast_data]

    # Run smc using the RatioHyperproperty (used to find speedup for 2 data sources)
    spa(data, RatioHyperproperty(), prob_threshold=0.5, confidence=0.9)


def test_insufficient_samples():
    # Get too little data to converge to a result
    slow_data = read_data(slow_data_file)[:10]

    with pytest.raises(SPAException):
        # Speed up testing by setting a low iteration limit
        spa(slow_data, ThresholdProperty(), prob_threshold=0.9, confidence=0.9, iteration_limit=5)
