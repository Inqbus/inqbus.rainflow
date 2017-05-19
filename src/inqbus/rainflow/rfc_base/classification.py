import numexpr as ne
import numpy as np


def classification(bin_count, array):
    """
    classifies 1d-array
    :param bin_count: Number of bins
    :param array: data to be classified
    :return: classified data as array
    """
    minimum = np.nanmin(array)
    maximum = np.nanmax(array)

    # categories are calculated by value * factor + summand and rounded
    factor = (1.0 - float(bin_count)) / (minimum - maximum)
    summand = -1.0 * minimum * factor + 1.0

    ex = 'value * factor + summand'

    classified_data = np.round(
        ne.evaluate(
            ex,
            local_dict={
                'value': array,
                'factor': factor,
                'summand': summand}),
        0)

    return classified_data
