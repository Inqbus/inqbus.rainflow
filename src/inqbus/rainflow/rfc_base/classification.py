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

    ex = 'value * (0.0 - (bin_count - 1.0)) / (minimum - maximum) +' + \
         '(-1.0 * minimum * (0.0 - (bin_count - 1.0)) / (minimum - maximum))'

    classified_data = np.round(
        ne.evaluate(
            ex,
            local_dict={
                'value': array,
                'bin_count': float(bin_count),
                'minimum': minimum,
                'maximum': maximum}),
        0)

    return classified_data


def binning_as_matrix(bin_count, array, minimum=None, maximum=None):
    """
    :param bin_count:
    :param array:
    :param maximum: maximum value to be recognized. Values bigger than max
    will be filtered
    :param minimum: minimum value to be recognized. Values smaller than min
    will be filtered
    :return: data matrix with start in rows and target in columns
    """

    # Bilding a 2d-histogram

    if not minimum:
        minimum = np.nanmin(array)
    if not maximum:
        maximum = np.nanmax(array)

    start = array[:, 0]
    target = array[:, 1]

    classified_data = np.histogram2d(start, target, bins=bin_count, range=[
                                     [minimum, maximum], [minimum, maximum]])

    return classified_data[0]
