import csv
from pathlib import Path
from typing import List

from spa.core import smc
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
    smc(slow_data, ThresholdProperty(threshold=4.5e11), prob_threshold=0.9, confidence=0.9)


def test_basic_ratio():
    # Get data
    slow_data = read_data(slow_data_file)
    fast_data = read_data(fast_data_file)

    # Prepare RatioHyperproperty data
    data = [slow_data, fast_data]

    # Run smc using the RatioHyperproperty (used to find speedup for 2 data sources)
    smc(data, RatioHyperproperty(threshold=0.9), prob_threshold=0.5, confidence=0.9)


def test_insufficient_samples():
    # Get too little
    slow_data = read_data(slow_data_file)[:10]

    result = smc(slow_data, ThresholdProperty(threshold=4.5e11), prob_threshold=0.9, confidence=0.9)

    assert result.result is None
