import csv
from pathlib import Path
from typing import List
import unittest

from spa.core import SPAException, spa
from spa.ThresholdProperty import ThresholdProperty
from spa.RatioHyperproperty import RatioHyperproperty
import pandas as pd

prop_1 = str(Path(__file__).parent) + '/data/prop1.csv'
prop_2 = str(Path(__file__).parent) + '/data/prop2.csv'
prop1_incomplete = str(Path(__file__).parent) + '/data/prop1_incomplete.csv'


def read_data(directory: str) -> List[int]:
    """Reads data from a csv file and returns it as a list of ints"""
    with open(directory, 'r') as file:
        reader = csv.reader(file, dialect='excel')
        string_vals = reader.__next__()

    data = [int(s) for s in string_vals]

    return data

def read_into_df(directory: str) -> pd.DataFrame:
    return pd.read_csv(directory)


def test_basic_threshold():
    # Find data to analyze
    data = read_into_df(prop_1)

    # Should complete successfully
    print(spa(data, ThresholdProperty(threshold=80), prob_threshold=0.9, confidence=0.9))


def test_basic_ratio():
    # Get data
    data = read_into_df(prop_1)

    # Run spa using the RatioHyperproperty (used to find speedup for 2 data sources)
    print(spa(data, RatioHyperproperty(threshold=0.6), prob_threshold=0.5, confidence=0.9))

def test_insufficient_samples():
    # Get too little data to converge to a result
    data = read_into_df(prop1_incomplete)

    with unittest.TestCase.assertRaises(None, SPAException):
        spa(data, ThresholdProperty(threshold=80), prob_threshold=0.9, confidence=0.9, iteration_limit=5)

test_basic_threshold()
test_basic_ratio()
test_insufficient_samples()