import os

import inqbus.rainflow as rfc
# Example for hdf5-file

# rainflow_on_hdf5_file:
# run base algorithm, store pairs and counted pairs to file

# classification_on_hdf5_file:
# classify results from rainflow_on_hdf5_file


testdatafile = 'testdata.h5'

# create_testdata

if not os.path.isfile(testdatafile):
    pass

# base algorithm
source_path = testdatafile + ':/testgroup/testdata'
source_column = 'value'
target_group = testdatafile + ':/statistics/testdata/value'

rfc.rainflow_on_hdf5_file(source_path, source_column, target_group)


# add some classification afterwards
source_path = target_group + '/RF_Pairs'

rfc.binning_on_hdf5_file(source_path,
                         target_group,
                         bin_count=32,
                         counted_table_name='RF_Counted_32',
                         pairs_table_name='RF_Pairs_32')

print('Calculation finished have a look at hdf5-file.')
