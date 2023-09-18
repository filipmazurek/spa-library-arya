import csv
from typing import List
import matplotlib.pyplot as plt

from spa.core import spa
from spa.properties import RatioHyperproperty, ThresholdProperty
from spa.util import min_num_samples


def read_data(directory: str) -> List[int]:
    """Reads data from a csv file and returns it as a list of ints"""
    with open(directory, 'r') as file:
        reader = csv.reader(file, dialect='excel')
        string_vals = reader.__next__()

    data = [int(s) for s in string_vals]

    return data


def hyperproperty_use():
    # Find data to analyze
    slow_data = read_data('data/ferret-l2_512kB-simticks.csv')
    fast_data = read_data('data/ferret-l2_3MB-simticks.csv')

    # For the sake of example, we are using only 22 samples, as done in the paper.
    # gem5 simulation with 4 OoO cores and Ruby memory system is expensive, so we minimize the number of samples taken
    slow_data = slow_data[:22]
    fast_data = fast_data[:22]

    # We will want to run SPA with a prob_threshold of 0.9 and confidence of 0.9. Check the minimum number of data
    #   points required for analysis
    min_samples = min_num_samples(prob_threshold=0.9, confidence=0.9)

    # Check that we have enough data
    assert len(slow_data) >= min_samples
    assert len(fast_data) >= min_samples

    # We will be using the ratio hyperproperty, so we need to combine the data into a single list
    data = [slow_data, fast_data]

    # Run SPA using the RatioHyperproperty (used to find speedup for 2 data sources)
    #   prob_threshold set at 0.5 (meaning finding the median value)
    #   with 90% confidence
    prob_threshold = 0.5
    confidence = 0.9
    result = spa(data, RatioHyperproperty(), prob_threshold=prob_threshold, confidence=confidence)

    low = result.confidence_interval.low
    high = result.confidence_interval.high

    # Plot the result using pyplot
    # Visualize the confidence intervals and true value
    fig, ax = plt.subplots()
    fig.subplots_adjust(top=0.855, left=0.135)

    # Plot the confidence intervals with caps
    cap_size = 5  # Cap size in points
    ax.errorbar([(low + high) / 2], [0], xerr=[[(high - low) / 2]], fmt='', capsize=cap_size, label='SPA')

    # Configure the plot
    ax.set_yticks([0])
    ax.set_yticklabels(['SPA'])
    ax.set_xlabel('Speedup')
    ax.set_title(
        f'SPA CI for Ferret Speedup\n512kB vs 3MB L2 Cache\n'
        f'N = {len(data[0])} | Proportion = {prob_threshold} | Confidence = {confidence}')
    ax.legend()

    # Show the plot
    plt.show()


def property_use():
    # Find data to analyze
    slow_data = read_data('data/ferret-l2_512kB-simticks.csv')
    fast_data = read_data('data/ferret-l2_3MB-simticks.csv')

    # For the sake of example, we are using only 22 samples, as done in the paper.
    slow_data = slow_data[:22]
    fast_data = fast_data[:22]

    # Create a list of speedups manually that will be used with the ThresholdProperty. This is equivalent to using the
    #   RatioHyperproperty, but is more simple. I recommend this method for understandability in this simple case.
    speedups = [slow / fast for slow, fast in zip(slow_data, fast_data)]

    # Run SPA using the ThresholdProperty to find the speedup confidence interval
    #   prob_threshold set at 0.5 (meaning finding the median value)
    #   with 90% confidence
    prob_threshold = 0.5
    confidence = 0.9
    result = spa(speedups, ThresholdProperty(), prob_threshold=prob_threshold, confidence=confidence)

    low = result.confidence_interval.low
    high = result.confidence_interval.high

    # Plot the result using pyplot
    # Visualize the confidence intervals and true value
    fig, ax = plt.subplots()
    fig.subplots_adjust(top=0.855, left=0.135)

    # Plot the confidence intervals with caps
    cap_size = 5  # Cap size in points
    ax.errorbar([(low + high) / 2], [0], xerr=[[(high - low) / 2]], fmt='', capsize=cap_size, label='SPA')

    # Configure the plot
    ax.set_yticks([0])
    ax.set_yticklabels(['SPA'])
    ax.set_xlabel('Speedup')
    ax.set_title(
        f'SPA CI for Ferret Speedup\n512kB vs 3MB L2 Cache\n'
        f'N = {len(speedups)} | Proportion = {prob_threshold} | Confidence = {confidence}')
    ax.legend()

    # Show the plot
    plt.show()


property_use()
