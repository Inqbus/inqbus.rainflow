import numexpr as ne
import numpy as np


def classification(number_of_classes, array):
    """
    classifies 1d-array
    :param number_of_classes: Number of bins
    :param array: data to be classified
    :return: classified data as array
    """
    minimum = np.nanmin(array)
    maximum = np.nanmax(array)

    # categories are calculated by value * factor + summand and rounded
    factor = (1 - number_of_classes) / (minimum - maximum)
    summand = -1 * minimum * factor + 1

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
