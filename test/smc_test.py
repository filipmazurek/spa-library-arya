import csv
from pathlib import Path
from typing import List

from spa.core import smc
from spa.ThresholdProperty import ThresholdProperty
from spa.RatioHyperproperty import RatioHyperproperty
from spa.Property2 import Property2
from spa.Property3 import Property3
from spa.Property4 import Property4
from spa.Property5 import Property5
from spa.Property6 import Property6
from spa.Property7 import Property7
import pandas as pd

prop_1 = str(Path(__file__).parent) + '/data/prop1.csv'
prop_2 = str(Path(__file__).parent) + '/data/prop2.csv'
prop_3 = str(Path(__file__).parent) + '/data/prop3.csv'
prop_4 = str(Path(__file__).parent) + '/data/prop4.csv'
prop_5 = str(Path(__file__).parent) + '/data/prop5.csv'
prop_6 = str(Path(__file__).parent) + '/data/prop6.csv'
prop_7 = str(Path(__file__).parent) + '/data/prop7.csv'
prop1_incomplete = str(Path(__file__).parent) + '/data/prop1_incomplete.csv'

def read_into_df(directory: str) -> pd.DataFrame:
    return pd.read_csv(directory)

def test_basic_threshold():
    # Find data to analyze
    data = read_into_df(prop_1)

    # Should complete successfully
    print(smc(data, ThresholdProperty(threshold=80), prob_threshold=0.9, confidence=0.9))


def test_basic_ratio():
    # Get data
    data = read_into_df(prop_1)

    # Run smc using the RatioHyperproperty (used to find speedup for 2 data sources)
    print(smc(data, RatioHyperproperty(threshold=0.6), prob_threshold=0.5, confidence=0.9))

def test_basic_prop2():
    # Get data
    data = read_into_df(prop_2)

    # Should complete successfully
    print(smc(data, Property2(threshold1 = 90, threshold2 = 150), prob_threshold=0.9, confidence=0.9))

def test_basic_prop3():
    data = read_into_df(prop_3)

    print(smc(data, Property3(threshold=0.2), prob_threshold=0.9, confidence=0.9))

def test_basic_prop4():
    data = read_into_df(prop_4)

    print(smc(data, Property4(threshold=95), prob_threshold=0.9, confidence=0.9))


def test_basic_prop5():
    data = read_into_df(prop_5)

    print(smc(data, Property5(threshold1=900, threshold2=5200), prob_threshold=0.9, confidence=0.9))

def test_basic_prop6():
    data = read_into_df(prop_6)

    print(smc(data, Property6(C=1000, threshold=0.6), prob_threshold=0.9, confidence=0.9))

def test_basic_prop7():
    data = read_into_df(prop_7)

    print(smc(data, Property7(threshold1=250, threshold2=600), prob_threshold=0.9, confidence=0.9))

def test_insufficient_samples():
    # Get too little data to converge to a result
    data = read_into_df(prop1_incomplete)
    result = smc(data, ThresholdProperty(threshold=80), prob_threshold=0.9, confidence=0.9)
    assert result.result is None

test_basic_threshold()
test_basic_ratio()
test_basic_prop2()
test_basic_prop3()
test_basic_prop4()
test_basic_prop5()
test_basic_prop6()
test_basic_prop7()
test_insufficient_samples()