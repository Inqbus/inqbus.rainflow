import numpy as np

import inqbus.rainflow as rfc

# Example for numpy array

# Rainflow.on_numpy_array:
# run base algorithm, return pairs and counted pairs

# Binning.as_table_on_numpy_array:
# classify results from Rainflow.on_numpy_array
# returns an array with pairs and an array with counted pairs as table

# Binning.as_matrix_on_numpy_array:
# classify results from Rainflow.on_numpy_array
# returns a 2d-array matrix like traditional rainflow matrix with
# start in rows and target in columns

# example data
N = 10

# data = np.random.random(N) * 5
data = np.array([ 4.22362955, 0.54226772, 4.74811879, 4.15082713, 0.39116187, 2.57925856, 1.79088452, 4.35611277, 0.34532837, 1.53295912])
print('==============Data==============')
print(data)

# main algorithm
res, res_counted = rfc.Rainflow.on_numpy_array(data)
print('==============Pairs==============')
print(res, res_counted)

# add some classifications afterwards
print('==============Binning as table==============')
res_32, res_counted_32 = rfc.Binning.as_table_on_numpy_array(res, bin_count=5)
print(res_32, res_counted_32)

print('==============Binning as Matrix==============')
matrix_32 = rfc.Binning.as_matrix_on_numpy_array(
    res, bin_count=5, axis=['top', 'left', 'bottom', 'right'])
print(matrix_32)

# add some classifications afterwards
print('==============Binning as table with min/max==============')
res_32, res_counted_32 = rfc.Binning.as_table_on_numpy_array(res, bin_count=5, minimum=0, maximum=5)

print(res_32, res_counted_32)

print('==============Binning as Matrix with min/max==============')
matrix_32 = rfc.Binning.as_matrix_on_numpy_array(
    res, bin_count=5, axis=['top', 'left', 'bottom', 'right'], minimum=0, maximum=5)

print(matrix_32)
