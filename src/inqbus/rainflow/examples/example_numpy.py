import numpy as np

import inqbus.rainflow as rfc

# Example for numpy array

# rainflow_on_numpy_array:
# run base algorithm, return pairs and counted pairs

# classification_on_numpy_array:
# classify results from rainflow_on_numpy_array

# example data
N = 1000000

data = np.random.random_integers(1, 1000, N)
print(data)

# main algorithm
res, res_counted = rfc.rainflow_on_numpy_array(data)
print(res, res_counted)

# add some classifications afterwards
res_32, res_counted_32 = rfc.binning_on_numpy_array(res, bin_count=32)
print(res_32, res_counted_32)