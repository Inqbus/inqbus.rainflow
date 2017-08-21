import unittest2 as unittest
import numpy as np
import inqbus.rainflow.rfc_base.classification as c


class TestClassification(unittest.TestCase):

    def test_binning_as_table(self):
        pairs = np.array(
            [
                [1.8, 1.7],
                [4.6, 0.0],
                [8.9, 3.7],
                [9.3, 1.3],
                [4.6, 8.2],
            ])

        res = c.binning(5, pairs)
        res2 = c.binning(5, pairs, remove_small_cycles=False)
        res3 = c.binning(
            5,
            pairs,
            minimum=1,
            maximum=5,
            remove_small_cycles=False)
        res4 = c.binning(5, pairs, minimum=1, maximum=5)

        self.assertTrue(np.array_equal([[3., 1.],
                                        [5., 2.],
                                        [5., 1.],
                                        [3., 5.]], res))

        self.assertTrue(np.array_equal([[1, 1],
                                        [3., 1.],
                                        [5., 2.],
                                        [5., 1.],
                                        [3., 5.]], res2))
        print(res3)
        self.assertTrue(np.array_equal([[2., 1.],
                                        [5., 0.], # zero because of not filtering values before
                                        [5., 4.],
                                        [5., 1.],
                                        [5., 5.]], res3))

        self.assertTrue(np.array_equal([[2., 1.],
                                        [5., 0.], # zero because of not filtering values before
                                        [5., 4.],
                                        [5., 1.]], res4))

    def test_binning_as_matrix(self):
        pairs = np.array(
            [
                [1.8, 1.7],
                [4.6, 0.0],
                [8.9, 3.7],
                [9.3, 1.3],
                [4.6, 8.2],
            ])

        res = c.binning_as_matrix(5, pairs)
        res2 = c.binning_as_matrix(5, pairs, remove_small_cycles=False)
        res3 = c.binning_as_matrix(
            5,
            pairs,
            minimum=1,
            maximum=5,
            remove_small_cycles=False)
        res4 = c.binning_as_matrix(5, pairs, minimum=1, maximum=5)

        self.assertTrue(np.array_equal([[0., 0., 0., 0., 0.],
                                        [0., 0., 0., 0., 0.],
                                        [1., 0., 0., 0., 1.],
                                        [0., 0., 0., 0., 0.],
                                        [1., 1., 0., 0., 0.]], res))

        self.assertTrue(np.array_equal([[1., 0., 0., 0., 0.],
                                        [0., 0., 0., 0., 0.],
                                        [1., 0., 0., 0., 1.],
                                        [0., 0., 0., 0., 0.],
                                        [1., 1., 0., 0., 0.]], res2))

        self.assertTrue(np.array_equal([[0., 0., 0., 0., 0.],
                                        [1., 0., 0., 0., 0.],
                                        [0., 0., 0., 0., 0.],
                                        [0., 0., 0., 0., 0.],
                                        [0., 0., 0., 0., 0.]], res3))

        self.assertTrue(np.array_equal([[0., 0., 0., 0., 0.],
                                        [1., 0., 0., 0., 0.],
                                        [0., 0., 0., 0., 0.],
                                        [0., 0., 0., 0., 0.],
                                        [0., 0., 0., 0., 0.]], res4))
