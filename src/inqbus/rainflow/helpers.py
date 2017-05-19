import numpy as np
import pandas as pd


def filter_data(data, minimum=None, maximum=None):
    if minimum is not None:
        data = data[np.where(data >= minimum)]
    if maximum is not None:
        data = data[np.where(data <= maximum)]
    return data


def get_extrema(data):

    a = np.diff(data)
    asign = np.sign(a)
    signchange = ((np.roll(asign, 1) - asign) != 0).astype(int)

    signchange[0] = 1
    np.append(signchange, [1])

    calc_data = data[np.where(signchange != 0)]

    return calc_data


def count_pairs(data):
    df = pd.DataFrame(data)

    a, b = df.columns.tolist()

    counts = df.groupby([a, b]).size()
    df = df.drop_duplicates()

    df[2] = pd.Series(counts.values, index=df.index)
    array = df.values
    return array
