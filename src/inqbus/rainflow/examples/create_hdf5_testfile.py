import numpy as np
import tables

print('Creating Testdata')
# Table schema


class Table(tables.IsDescription):
    value = tables.Float64Col(pos=1)

# Testdata


N = 1000000

data = np.round(np.random.random(N), 2)

# Tablesettings

filename = 'testdata.h5'
table_group = 'testgroup'
table_name = 'testdata'

# creation

hdf5_file = tables.open_file(filename, mode='a')
try:
    hdf5_file.create_group('/', table_group, table_group)
except tables.exceptions.NodeError:
    pass
table = hdf5_file.create_table(
    '/' + table_group,
    table_name,
    Table,
    table_name,
)
table.append(data)
table.flush()
hdf5_file.close()

print('Finished Creating Testdata')
