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

        self.assertTrue(np.array_equal([[2., 0.],
                                        [4., 1.],
                                        [4., 0.],
                                        [2., 4.]], res))

        self.assertTrue(np.array_equal([[0, 0],
                                        [2., 0.],
                                        [4., 1.],
                                        [4., 0.],
                                        [2., 4.]], res2))
        self.assertTrue(np.array_equal([[1., 0.],
                                        [4., -1.], # -1 because of not filtering values before
                                        [4., 3.],
                                        [4., 0.],
                                        [4., 4.]], res3))

        self.assertTrue(np.array_equal([[1., 0.],
                                        [4., -1.], # -1 because of not filtering values before
                                        [4., 3.],
                                        [4., 0.]], res4))

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
