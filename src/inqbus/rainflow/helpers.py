import numpy as np
import pandas as pd


def filter_outliers(data, minimum=None, maximum=None):
    if minimum is not None:
        data = data[np.where(data >= minimum)]
    if maximum is not None:
        data = data[np.where(data <= maximum)]
    return data


def filter_outliers_on_pairs(data, minimum=None, maximum=None):
    if minimum is not None:
        data = data[data[:, 1] >= minimum, :]
        data = data[data[:, 0] >= minimum, :]

    if maximum is not None:
        data = data[data[:, 1] <= maximum, :]
        data = data[data[:, 0] <= maximum, :]
    return data


def get_extrema(data):
    # find extrema by finding indexes where diff changes sign
    data_diff = np.diff(data)
    asign = np.sign(data_diff)
    signchange = ((np.roll(asign, 1) - asign) != 0).astype(int)

    # first and last value is always a local extrema
    signchange[0] = 1

    # last value is missing because the diff-array is 1 value shorter than the
    # input array so we have to add it again
    signchange = np.append(signchange, np.array([1]))

    calc_data = data[np.where(signchange != 0)]

    return calc_data


def count_pairs(data):
    df = pd.DataFrame(data)

    start, target = df.columns.tolist()

    # first we create groups for each pair and take size of each group as count.
    # counts is a pandas.Series with the pairs as index
    counts = df.groupby([start, target]).size()

    # than we remove duplicate pairs from original dateframe,
    # so length and counts are equal in size
    df = df.drop_duplicates()

    # reset index to values of pairs to fit index of counts
    df.set_index([0, 1], inplace=True, drop=False)

    # now we append the counts as column to the original data
    df[2] = pd.Series(counts.values, index=counts.index)

    # just cast pandas-dataframe back to numpy 2d-array usable for following
    # steps
    array = df.values
    return array


def append_axis(data, horizontal_axis=None, vertical_axis=None,
                axis=[]):
    """
    :param vertical_axis: numpy 1-d-array,
        length must be equal to number of matrix columns
    :param horizontal_axis: numpy 1-d-array,
        length must be equal to number of matrix rows
    :param data: matrix where axis should be added
    :param axis: list of placements for axis. Possible list-elements are:
            * 'bottom'
            * 'left'
            * 'right'
            * 'top'

    :return: matrix (2d-array) with axis
    """
    matrix_shape = data.shape
    horizontal_axis_shape = horizontal_axis.shape
    vertical_axis_shape = vertical_axis.shape

    vertical_shape_fit = matrix_shape[1] == vertical_axis_shape[0]
    horizontal_shape_fit = matrix_shape[0] == horizontal_axis_shape[0]

    if 'top' in axis and horizontal_axis is not None and horizontal_shape_fit:
        # vertical axis increases length by one in the upper corner
        vertical_axis = np.append(np.NaN, vertical_axis)
        # append upper axis
        data = np.vstack((horizontal_axis, data))
    if 'bottom' in axis and horizontal_axis is not None and horizontal_shape_fit:
        # vertical axis increases length by one in the bottom corner
        vertical_axis = np.append(vertical_axis, np.NaN)
        # append bottom axis
        data = np.vstack((data, horizontal_axis))

    # reshape to a column
    vertical_axis = vertical_axis.reshape((vertical_axis.size, 1))

    if 'left' in axis and vertical_axis is not None and vertical_shape_fit:
        # append on left sids
        data = np.hstack((vertical_axis, data))
    if 'right' in axis and vertical_axis is not None and vertical_shape_fit:
        # append on right site
        data = np.hstack((data, vertical_axis))

    return data
