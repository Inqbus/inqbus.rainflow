import pyximport
pyximport.install()
import numpy as np

from inqbus.rainflow.data_sources.hdf5 import HDF5Table, RFCTable, \
    RFCCountedTable
from inqbus.rainflow.rfc_base.rainflow import rainflow
from inqbus.rainflow.rfc_base.classification import binning, binning_as_matrix
from inqbus.rainflow.helpers import filter_outliers, get_extrema, count_pairs, \
    filter_outliers_on_pairs, append_axis


class Rainflow(object):

    @staticmethod
    def on_numpy_array(
            data,
            maximum=None,
            minimum=None,
            bin_count=64,
            classify=False):
        """
        :param classify: True if classification should be done
        :param data: input data, 1d-numpy-array
        :param maximum: maximum value to be recognized. Values bigger than max
        will be filtered
        :param minimum: minimum value to be recognized. Values smaller than min
        will be filtered
        :param bin_count: integer for classification. Describes number of classes
        :return: array with pairs, array with counted pairs

        2d-array pairs with data like (start, target)
        2d-array counted pairs with data like (start, target, count)
        """
        if minimum or maximum:
            data = filter_outliers(data, minimum=minimum, maximum=maximum)

        if classify:
            data = binning(bin_count, data)

        local_extrema = get_extrema(data)

        result_pairs, residuen_vector = rainflow(local_extrema)

        result_counted = count_pairs(result_pairs)

        return result_pairs, result_counted

    @staticmethod
    def on_hdf5_file(source_table,
                     source_column,
                     target_group,
                     counted_table_name='RF_Counted',
                     pairs_table_name='RF_Pairs',
                     maximum=None,
                     minimum=None,
                     bin_count=64,
                     classify=False):
        """
        :param classify: True if classification should be done
        :param pairs_table_name: Table name for storing Pairs
        :param counted_table_name: Table name for storing Counted Pairs
        :param source_table: hdf5-url for table where data should be read
        :param source_column: name of column which should be used
        :param target_group: hdf5-url where to store data
        :param maximum: maximum value to be recognized. Values bigger than max
        will be filtered
        :param minimum: minimum value to be recognized. Values smaller than min
        will be filtered
        :param bin_count: integer for classification. Describes number of classes
        :return:
        """
        source_table_obj = HDF5Table(source_table)

        data = source_table_obj.read_from_file(source_column)

        source_table_obj.close()

        result_pairs, result_counted = Rainflow.on_numpy_array(
            data,
            minimum=minimum,
            maximum=maximum,
            bin_count=bin_count,
            classify=classify
        )

        table_path_pairs = '/'.join([target_group, pairs_table_name])
        table_path_counted = '/'.join([target_group, counted_table_name])

        pairs_table = HDF5Table(
            table_path_pairs,
            table_class=RFCTable,
            create_empty_table=True)
        pairs_table.write_to_file(result_pairs)
        pairs_table.close()

        counted_table = HDF5Table(
            table_path_counted,
            table_class=RFCCountedTable,
            create_empty_table=True)
        counted_table.write_to_file(result_counted)
        counted_table.close()


class Binning(object):

    @staticmethod
    def as_table_on_numpy_array(
            array,
            bin_count=64,
            maximum=None,
            minimum=None,
            remove_small_cycles=True):
        """
        Use this to add a classification after running the rainflow algorithm

        :param remove_small_cycles: if True cycles where start and end are
        identical after binning will be removed
        :param array: result array with pairs like returned from
        rainflow_on_numpy_array; 2d-array with data like (start, target)
        :param bin_count: number of classes
        :param maximum: maximum value to be recognized. Values bigger than max
        will be filtered
        :param minimum: minimum value to be recognized. Values smaller than min
        will be filtered
        :return:result array with pairs, counted result array
        """
        if minimum or maximum:
            array = filter_outliers_on_pairs(
                array,
                minimum=minimum,
                maximum=maximum)

        res_pairs = binning(
            bin_count,
            array,
            remove_small_cycles=remove_small_cycles,
            minimum=minimum,
            maximum=maximum)
        res_counted = count_pairs(res_pairs)

        return res_pairs, res_counted

    @staticmethod
    def as_matrix_on_numpy_array(
            array,
            bin_count=64,
            maximum=None,
            minimum=None,
            axis=[],
            remove_small_cycles=True):
        """
        Use this to get classified data as matrix after rainflow-algorithm

        :param remove_small_cycles: if True cycles where start and end are
        identical after binning will be removed
        :param array: result array with pairs like returned from
        rainflow_on_numpy_array; 2d-array with data like (start, target)
        :param bin_count: number of classes
        :param maximum: maximum value to be recognized. Values bigger than max
        will be filtered
        :param minimum: minimum value to be recognized. Values smaller than min
        will be filtered
        :param axis: list of placements for axis. Possible list-elements are:
            * 'bottom'
            * 'left'
            * 'right'
            * 'top'
        :return:result array with shape (bin_count, bin_count) showing the count of
        pairs as 2d-histo with start in rows and target in columns
        """

        res_matrix = binning_as_matrix(
            bin_count,
            array,
            minimum=minimum,
            maximum=maximum,
            axis=axis,
            remove_small_cycles=remove_small_cycles)

        return res_matrix

    @staticmethod
    def as_table_on_hdf5_file(
            source_table,
            target_group,
            bin_count=64,
            counted_table_name='RF_Counted_64',
            pairs_table_name='RF_Pairs_64',
            maximum=None,
            minimum=None,
            remove_small_cycles=True):
        """
        Use this to add a classification after running the rainflow algorithm

        :param remove_small_cycles: if True cycles where start and end are
        identical after binning will be removed
        :param source_table: Table which includes pairs. Should be table like
        created in rainflow_for_hdf5
        :param target_group: hdf5-url where to store data
        :param bin_count: number of classes
        :param pairs_table_name: Table name for storing Pairs
        :param maximum: maximum value to be recognized. Values bigger than max
        will be filtered
        :param minimum: minimum value to be recognized. Values smaller than min
        will be filtered
        :param counted_table_name: Table name for storing Counted Pairs

        """
        source_table_ob = HDF5Table(source_table)
        start = source_table_ob.read_from_file('start')
        target = source_table_ob.read_from_file('target')
        source_table_ob.close()

        data = np.stack((start, target), axis=-1)

        result_pairs, result_counted = Binning.as_table_on_numpy_array(
            data,
            bin_count=bin_count,
            minimum=minimum,
            maximum=maximum,
            remove_small_cycles=remove_small_cycles
        )

        table_path_pairs = '/'.join([target_group, pairs_table_name])
        table_path_counted = '/'.join([target_group, counted_table_name])

        pairs_table = HDF5Table(
            table_path_pairs,
            table_class=RFCTable,
            create_empty_table=True)
        pairs_table.write_to_file(result_pairs)
        pairs_table.close()

        counted_table = HDF5Table(
            table_path_counted,
            table_class=RFCCountedTable,
            create_empty_table=True)
        counted_table.write_to_file(result_counted)
        counted_table.close()

    @staticmethod
    def as_matrix_on_hdf5_file(
            source_table,
            target_group,
            bin_count=64,
            counted_table_name='RF_Matrix_64',
            maximum=None,
            minimum=None,
            axis=[],
            remove_small_cycles=True):
        """
        Use this to add a classification after running the rainflow algorithm

        :param remove_small_cycles: if True cycles where start and end
        are identical after binning will be removed
        :param axis: list of placements for axis. Possible list-elements are:
            * 'bottom'
            * 'left'
            * 'right'
            * 'top'
        :param source_table: Table which includes pairs. Should be table like
        created in rainflow_for_hdf5
        :param target_group: hdf5-url where to store data
        :param bin_count: number of classes
        :param pairs_table_name: Table name for storing Pairs
        :param maximum: maximum value to be recognized. Values bigger than max
        will be filtered
        :param minimum: minimum value to be recognized. Values smaller than min
        will be filtered
        :param counted_table_name: Table name for storing Counted Pairs

        stores result array with shape (bin_count, bin_count) showing the count of
        pairs as 2d-histo with start in rows and target in columns
        """
        source_table_ob = HDF5Table(source_table)
        start = source_table_ob.read_from_file('start')
        target = source_table_ob.read_from_file('target')
        source_table_ob.close()

        data = np.stack((start, target), axis=-1)

        result_matrix = Binning.as_matrix_on_numpy_array(
            data,
            bin_count=bin_count,
            minimum=minimum,
            maximum=maximum,
            axis=axis,
            remove_small_cycles=remove_small_cycles
        )

        table_path_counted = '/'.join([target_group, counted_table_name])

        pairs_table = HDF5Table(
            table_path_counted,
            table_class='',
            create_empty_table=False)
        pairs_table.write_array_to_file(result_matrix)
        pairs_table.close()
