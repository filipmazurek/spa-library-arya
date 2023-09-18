import csv
from typing import List

from spa.core import smc
from spa.properties import ThresholdProperty


def read_data(directory: str) -> List[int]:
    """Reads data from a csv file and returns it as a list of ints"""
    with open(directory, 'r') as file:
        reader = csv.reader(file, dialect='excel')
        string_vals = reader.__next__()

    data = [int(s) for s in string_vals]

    return data


# Find data to analyze
data = read_data('data/ferret-l2_512kB-simticks.csv')

# Run SPA using the ThresholdProperty. The threshold is set to 4.5e11 cycles. Meaning the property is satsified if the
#   data is faster than 4.5e11 cycles.
#   The prob_threshold is set to 0.9, meaning that we want 90% of the population to satisfy this threshold criteria
#   The confidence is set to 0.9, meaning that we want 90% confidence in the result
result = smc(data, ThresholdProperty(threshold=4.5e11), prob_threshold=0.9, confidence=0.9)

print(result)
