import pyximport
pyximport.install()
import numpy as np

from inqbus.rainflow.data_sources.hdf5 import HDF5Table, RFCTable, \
    RFCCountedTable
from inqbus.rainflow.rfc_base.rainflow import rainflow
from inqbus.rainflow.rfc_base.classification import classification
from inqbus.rainflow.helpers import filter_outliers, get_extrema, count_pairs


def rainflow_on_numpy_array(
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
        data = classification(bin_count, data)

    local_extrema = get_extrema(data)

    result_pairs, residuen_vector = rainflow(local_extrema)

    result_counted = count_pairs(result_pairs)

    return result_pairs, result_counted


def classification_on_numpy_array(array, bin_count=64):
    """
    Use this to add a classification after running the rainflow algorithm

    :param array: result array with pairs like returned from
    rainflow_on_numpy_array; 2d-array with data like (start, target)
    :param bin_count: number of classes
    :return:result array with pairs, counted result array
    """
    res_pairs = classification(bin_count, array)
    res_counted = count_pairs(res_pairs)

    return res_pairs, res_counted


def rainflow_on_hdf5_file(source_table,
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

    result_pairs, result_counted = rainflow_on_numpy_array(
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


def classification_on_hdf5_file(source_table,
                                target_group,
                                bin_count=64,
                                counted_table_name='RF_Counted_64',
                                pairs_table_name='RF_Pairs_64'):
    """
    Use this to add a classification after running the rainflow algorithm

    :param source_table: Table which includes pairs. Should be table like
    created in rainflow_for_hdf5
    :param target_group: hdf5-url where to store data
    :param bin_count: number of classes
    :param pairs_table_name: Table name for storing Pairs
    :param counted_table_name: Table name for storing Counted Pairs
    :return:
    """
    source_table_ob = HDF5Table(source_table)
    start = source_table_ob.read_from_file('start')
    target = source_table_ob.read_from_file('target')
    source_table_ob.close()

    data = np.stack((start, target), axis=-1)

    result_pairs, result_counted = classification_on_numpy_array(
        data,
        bin_count=bin_count
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
