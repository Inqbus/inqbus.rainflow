import numpy as np

from inqbus.rainflow.run_rainflow import rainflow_for_numpy, \
    classification_for_numpy

# example data
N = 1000000

data = np.random.random_integers(1, 1000, N)
print(data)

# main algorithm
res, res_counted = rainflow_for_numpy(data)
print(res, res_counted)

# add some classifications afterwards
res_32, res_counted_32 = classification_for_numpy(res, number_of_classes=32)
print(res_32, res_counted_32)
