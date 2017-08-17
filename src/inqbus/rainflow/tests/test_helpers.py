import unittest2 as unittest
import numpy as np
import inqbus.rainflow.helpers as h


class TestHelpers(unittest.TestCase):

    def test_filter_outliers(self):
        data = np.array([1.0, 2.0, 3.0, 4.0])

        res_min = h.filter_outliers(data, minimum=2.0)
        res_max = h.filter_outliers(data, maximum=2.0)
        res_no_outliers = h.filter_outliers(data)
        res_both = h.filter_outliers(data, minimum=2.0, maximum=3.0)

        self.assertTrue(np.array_equal(res_min, np.array([2.0, 3.0, 4.0])))
        self.assertTrue(np.array_equal(res_max, np.array([1.0, 2.0])))
        self.assertTrue(np.array_equal(res_no_outliers, data))
        self.assertTrue(np.array_equal(res_both, np.array([2.0, 3.0])))

    def test_filter_outliers_on_pairs(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0], [2.0, 1.0]])

        res_min = h.filter_outliers_on_pairs(data, minimum=2.0)
        res_max = h.filter_outliers_on_pairs(data, maximum=2.0)
        res_no_outliers = h.filter_outliers_on_pairs(data)
        res_both = h.filter_outliers_on_pairs(data, minimum=2.0, maximum=4.0)

        self.assertTrue(np.array_equal(res_min, np.array([[3.0, 4.0]])))
        self.assertTrue(np.array_equal(
            res_max, np.array([[1.0, 2.0], [2.0, 1.0]])))
        self.assertTrue(np.array_equal(res_no_outliers, data))
        self.assertTrue(np.array_equal(res_both, np.array([[3.0, 4.0]])))

    def test_get_extrema(self):
        data = np.array([1, 3, 4, 7, 6, 8, 2])

        res = h.get_extrema(data)

        self.assertTrue(np.array_equal(res, np.array([1, 7, 6, 8, 2])))

    def test_count_pairs(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0], [2.0, 1.0]])

        res = h.count_pairs(data)

        self.assertTrue(np.array_equal(res, np.array(
            [[1.0, 2.0, 1.0], [3.0, 4.0, 1.0], [2.0, 1.0, 1.0]])))

        data = np.array([[1.0, 2.0], [3.0, 4.0], [1.0, 2.0]])

        res = h.count_pairs(data)

        self.assertTrue(np.array_equal(
            res, np.array([[1.0, 2.0, 2.0], [3.0, 4.0, 1.0]])))

    def test_append_axis(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        axis = np.array([0, 5])

        res1 = h.append_axis(
            data,
            horizontal_axis=axis,
            vertical_axis=axis,
            axis=[])
        res2 = h.append_axis(
            data,
            horizontal_axis=axis,
            vertical_axis=axis,
            axis=['bottom'])
        res3 = h.append_axis(
            data,
            horizontal_axis=axis,
            vertical_axis=axis,
            axis=['top'])
        res4 = h.append_axis(
            data,
            horizontal_axis=axis,
            vertical_axis=axis,
            axis=['left'])
        res5 = h.append_axis(
            data,
            horizontal_axis=axis,
            vertical_axis=axis,
            axis=['right'])
        res6 = h.append_axis(
            data,
            horizontal_axis=axis,
            vertical_axis=axis,
            axis=['blub'])

        self.assertTrue(np.array_equal(res1, data))
        self.assertTrue(np.array_equal(
            res2, np.array([[1, 2], [3, 4], [0, 5]])))
        self.assertTrue(np.array_equal(
            res3, np.array([[0, 5], [1, 2], [3, 4]])))
        self.assertTrue(np.array_equal(res4, np.array([[0, 1, 2], [5, 3, 4]])))
        self.assertTrue(np.array_equal(res5, np.array([[1, 2, 0], [3, 4, 5]])))
        self.assertTrue(np.array_equal(res6, data))
