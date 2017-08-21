import numexpr as ne
import numpy as np

from inqbus.rainflow.helpers import append_axis


def binning(
        bin_count,
        array,
        remove_small_cycles=True,
        minimum=None,
        maximum=None,
        count_from_zero=False):
    """
    classifies array
    :param count_from_zero: set TRue when classes should be counted from 0, else
        they start on 1
    :param maximum: maximum value to be recognized. Values bigger than max
        will be put in highest class. So filter before
    :param minimum: minimum value to be recognized. Values smaller than min
        will added to smallest class. So filter before
    :param bin_count: Number of bins
    :param array: data to be classified
    :param remove_small_cycles: if True cycles where start and end are
        identical after binning will be removed
    :return: classified data as array
    """
    if minimum is None:
        minimum = np.nanmin(array)
    if maximum is None:
        maximum = np.nanmax(array)

    minimum = float(minimum)
    maximum = float(maximum)

    bin_width = ((maximum - minimum) / bin_count)

    bin_borders = []
    bin = minimum
    classes = range(0, bin_count)

    for x in classes:
        bin_borders.append(bin)
        bin = bin + bin_width



    # fit values to classes 0 .. bin_count
    if count_from_zero:
        classified_data = np.digitize(array, bin_borders) - 1.0
    else:
        classified_data = np.digitize(array, bin_borders)

    if remove_small_cycles:
        diff_of_cols = classified_data[:, 0] - classified_data[:, 1]
        classified_data = classified_data[np.nonzero(diff_of_cols)]

    return classified_data


def binning_as_matrix(
        bin_count,
        array,
        minimum=None,
        maximum=None,
        axis=[],
        remove_small_cycles=True):
    """
    :param bin_count:
    :param array:
    :param maximum: maximum value to be recognized. Values bigger than max
    will be filtered
    :param minimum: minimum value to be recognized. Values smaller than min
    will be filtered
    :param axis: list of placements for axis. Possible list-elements are:
            * 'bottom'
            * 'left'
            * 'right'
            * 'top'
    :param remove_small_cycles: if True cycles where start and end are
        identical after binning will be removed
    :return: data matrix with start in rows and target in columns
    """

    # Bilding a 2d-histogram

    if minimum is None:
        minimum = np.nanmin(array)
    if maximum is None:
        maximum = np.nanmax(array)

    minimum = float(minimum)
    maximum = float(maximum)

    start = array[:, 0]
    target = array[:, 1]

    classified_data = np.histogram2d(start, target, bins=bin_count, range=[
                                     [minimum, maximum], [minimum, maximum]])

    res_matrix = classified_data[0]
    intervall_edges = classified_data[1]

    axis_value = np.diff(intervall_edges) / 2.0 + intervall_edges[0:-1]

    if remove_small_cycles:
        indexes = np.diag_indices(bin_count)
        res_matrix[indexes] = 0.0

    if axis:
        res_matrix = append_axis(res_matrix, horizontal_axis=axis_value,
                                 vertical_axis=axis_value, axis=axis)

    return res_matrix
