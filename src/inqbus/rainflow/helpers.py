import numpy as np
import pandas as pd


def filter_outliers(data, minimum=None, maximum=None):
    if minimum is not None:
        data = data[np.where(data >= minimum)]
    if maximum is not None:
        data = data[np.where(data <= maximum)]
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

    # now we append the counts as column to the original data
    df[2] = pd.Series(counts.values, index=df.index)

    # just cast pandas-dataframe back to numpy 2d-array usable for following
    # steps
    array = df.values
    return array
