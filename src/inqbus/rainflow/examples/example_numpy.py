import numpy as np

import inqbus.rainflow as rfc

# Example for numpy array

# rainflow_on_numpy_array:
# run base algorithm, return pairs and counted pairs

# binning_as_table_on_numpy_array:
# classify results from rainflow_on_numpy_array
# returns an array with pairs and an array with counted pairs as table

# binning_as_matrix_on_numpy_array:
# classify results from rainflow_on_numpy_array
# returns a 2d-array matrix like traditional rainflow matrix with
# start in rows and target in columns

# example data
N = 1000000

data = np.random.random(N) * 1000
print(data)

# main algorithm
res, res_counted = rfc.rainflow_on_numpy_array(data)
print(res, res_counted)

# add some classifications afterwards
res_32, res_counted_32 = rfc.binning_as_table_on_numpy_array(res, bin_count=32)
print(res_32, res_counted_32)

matrix_32 = rfc.binning_as_matrix_on_numpy_array(res, bin_count=32)
print(matrix_32)
