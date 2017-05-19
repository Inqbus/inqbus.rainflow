import os

from inqbus.rainflow.run_rainflow import rainflow_for_hdf5, \
    classification_for_hdf5


testdatafile = 'testdata.h5'

# create_testdata

if not os.path.isfile(testdatafile):
    pass

# base algorithm
source_path = testdatafile + ':/testgroup/testdata'
source_column = 'value'
target_group = testdatafile + ':/statistics/testdata/value'

rainflow_for_hdf5(source_path, source_column, target_group)


# add some classification afterwards
source_path = target_group + '/RF_Pairs'

classification_for_hdf5(source_path,
                        target_group,
                        number_of_classifications=32,
                        counted_table_name='RF_Counted_32',
                        pairs_table_name='RF_Pairs_32')

print('Calculation finished have a look at hdf5-file.')
