import tables
import pandas as pd


class RFCTable(tables.IsDescription):
    start = tables.Float64Col(pos=1)  # @UndefinedVariable
    target = tables.Float64Col(pos=2)  # @UndefinedVariable


class RFCCountedTable(tables.IsDescription):
    start = tables.Float64Col(pos=1)  # @UndefinedVariable
    target = tables.Float64Col(pos=2)  # @UndefinedVariable
    count = tables.Float64Col(pos=3)


class MissingTableClass(Exception):
    pass


class HDF5Table(object):

    def __init__(self, path,
                 table_class=None,
                 create_empty_table=False,
                 mode='a'):
        self.table = None
        self.file = None
        self.path = path
        self.file_path, self.table_path = self.path.split(':')

        # make sure all table_paths start with a single '/'
        self.table_path = '/' + self.table_path.strip('/')
        self.mode = mode
        self.open_file()
        self.get_table(table_class, create_empty_table)

    def open_file(self):
        self.file = tables.open_file(self.file_path, mode=self.mode)

    def get_table(self, table_class, create_empty_table):
        if create_empty_table and table_class:
            self.remove()
            self.create_empty_table(table_class)
        elif create_empty_table:
            raise MissingTableClass()
        else:
            self.find_table()

    def close(self):
        if self.table:
            self.table.flush()
        self.file.close()

    def read_from_file(self, column):
        table_data = pd.DataFrame.from_records(self.table.read())

        data = table_data[column].values
        return data

    def write_to_file(self, array):
        self.table.append(array)
        self.table.flush()

    def create_empty_table(self, table_class):
        path_elements = self.table_path.split('/')
        table_name = path_elements.pop()
        node_path = '/'.join(path_elements) + '/'

        path = ""
        for path_element in path_elements:
            old_path = path
            path += path_element + "/"
            try:
                self.file.get_node(path, "")
            except tables.NoSuchNodeError:
                self.file.create_group(old_path, path_element,
                                       path_element)

        self.table = self.file.create_table(
            node_path,
            table_name,
            table_class,
            table_name,
        )

    def remove(self):
        path_elements = self.table_path.split('/')
        table_name = path_elements.pop()
        node_path = '/'.join(path_elements) + '/'
        try:
            self.file.remove_node(node_path, name=table_name)
        except tables.exceptions.NoSuchNodeError:
            pass
        self.table = None

    def find_table(self):
        path_elements = self.table_path.split('/')
        table_name = path_elements.pop()
        node_path = '/'.join(path_elements) + '/'
        self.table = self.file.get_node(node_path, table_name)
