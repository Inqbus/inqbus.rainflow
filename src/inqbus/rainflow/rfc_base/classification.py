import numexpr as ne
import numpy as np


def binning(bin_count, array):
    """
    classifies array
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


def binning_as_matrix(bin_count, array):
    """
    :param bin_count:
    :param array:
    :return: data matrix with start in rows and target in columns
    """

    # fit data to number of bins if this is not done, data outside bin-range
    # would be ignored
    binned_data = binning(bin_count, array)

    start = binned_data[:, 0]
    target = binned_data[:, 1]
    bins = range(1, bin_count + 1)

    classified_data = np.histogram2d(start, target, bins=bins)

    return classified_data[0]
