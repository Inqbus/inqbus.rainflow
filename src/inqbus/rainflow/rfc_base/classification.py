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
    factor = (0.0 - (float(bin_count) - 1)) / (minimum - maximum)
    summand = -1.0 * minimum * factor

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

    # Bilding a 2d-histogram

    minimum = np.nanmin(array)
    maximum = np.nanmax(array)

    start = array[:, 0]
    target = array[:, 1]

    classified_data = np.histogram2d(start, target, bins=bin_count, range=[
                                     [minimum, maximum], [minimum, maximum]])

    return classified_data[0]
