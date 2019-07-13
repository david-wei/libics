import unittest
import numpy as np

from libics.util import misc


###############################################################################


class StringFunctionsTestCase(unittest.TestCase):

    def test_letter_range(self):
        self.assertEqual(list(misc.letter_range("d")), ["a", "b", "c", "d"])
        self.assertEqual(list(misc.letter_range(4)), ["a", "b", "c", "d"])
        self.assertEqual(list(misc.letter_range("b", "d")), ["b", "c", "d"])
        self.assertEqual(list(misc.letter_range("b", 3)), ["b", "c", "d"])
        self.assertEqual(list(misc.letter_range("a", "d", 2)), ["a", "c"])
        self.assertEqual(list(misc.letter_range("e", "a")), [])


###############################################################################


class ArrayFunctionsTestCase(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_vectorize_tensorize(self):
        ar = np.arange(120).reshape((2, 3, 4, 5))
        tensor_axes = (0, 2)
        vec_axis = 2
        vec, vec_shape = misc.vectorize_numpy_array(
            ar, tensor_axes=tensor_axes, vec_axis=vec_axis, ret_shape=True
        )
        res_ar = misc.tensorize_numpy_array(
            vec, vec_shape, tensor_axes=tensor_axes, vec_axis=vec_axis
        )
        self.assertTrue(np.allclose(ar, res_ar))

    def test_tensormul_tensorinv(self):
        # Matrix and broadcasting test
        ar = np.sqrt(np.arange(2 * 3 * 3)).reshape((2, 3, 3))
        inv_ar = misc.tensorinv_numpy_array(ar, a_axes=1, b_axes=2)
        res_id = misc.tensormul_numpy_array(
            ar, inv_ar, a_axes=(0, 1, 2), b_axes=(0, 2, 3), res_axes=(0, 1, 3)
        )
        self.assertTrue(np.all(np.isclose(res_id, 0) | np.isclose(res_id, 1)))
        # Tensor dot and broadcasting test
        ar = np.sqrt(np.arange(2 * 2 * 2 * 3 * 3)).reshape((2, 2, 2, 3, 3))
        print("+++++++ AR ++++++")
        print(ar)
        inv_ar = misc.tensorinv_numpy_array(ar, a_axes=(0, 3), b_axes=(1, 4))
        print("+++++++ INV_AR ++++++")
        print(inv_ar)
        res_id = misc.tensormul_numpy_array(
            ar, inv_ar,
            a_axes=(0, 1, 2, 3, 4), b_axes=(1, 5, 2, 4, 6),
            res_axes=(0, 5, 2, 3, 6)
        )
        print("+++++++ RES_ID ++++++")
        print(res_id)
        prod = misc.tensormul_numpy_array(
            res_id, ar,
            a_axes=(0, 1, 2, 3, 4), b_axes=(1, 5, 2, 4, 6),
            res_axes=(0, 5, 2, 3, 6)
        )
        print("+++++++ DIFF ++++++++")
        print(prod - ar)
        self.assertTrue(np.all(np.isclose(res_id, 0) | np.isclose(res_id, 1)))


###############################################################################


if __name__ == '__main__':
    unittest.main()
