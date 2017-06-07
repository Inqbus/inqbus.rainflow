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
        data = data[:, data[0] >= minimum]
        data = data[:, data[1] >= minimum]
    if maximum is not None:
        data = data[:, data[0] <= maximum]
        data = data[:, data[1] <= maximum]
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
    np.append(signchange, [1])

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

def append_axis(data, bin_count, axis=[]):
    """
    :param data: matrix where axis should be added
    :param bin_count: numer of bins
    :param axis: list of placements for axis. Possible list-elements are:
            * 'bottom'
            * 'left'
            * 'right'
            * 'top'
    :return: matrix (2d-array) with axis
    """
    horizontal_axis = np.array(range(0, bin_count)) * 1.0
    vertical_axis = np.array(range(0, bin_count)) * 1.0

    if 'top' in axis:
        # vertical axis increases length by one in the upper corner
        vertical_axis = np.append(np.NaN, vertical_axis)
        # append upper axis
        data = np.vstack((horizontal_axis, data))
    if 'bottom' in axis:
        # vertical axis increases length by one in the bottom corner
        vertical_axis = np.append(vertical_axis, np.NaN)
        # append bottom axis
        data = np.vstack((data, horizontal_axis))

    # reshape to a column
    vertical_axis = vertical_axis.reshape((vertical_axis.size, 1))

    if 'left' in axis:
        # append on left sids
        data = np.hstack((vertical_axis, data))
    if 'right' in axis:
        # append on right site
        data = np.hstack((data, vertical_axis))


    return data