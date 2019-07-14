import unittest
import numpy as np

from libics.util import misc


###############################################################################


class StringFunctionsTestCase(unittest.TestCase):

    def test_char_range(self):
        self.assertEqual(list(misc.char_range("d")), ["a", "b", "c", "d"])
        self.assertEqual(list(misc.char_range(4)), ["a", "b", "c", "d"])
        self.assertEqual(list(misc.char_range("b", "d")), ["b", "c", "d"])
        self.assertEqual(list(misc.char_range("b", 3)), ["b", "c", "d"])
        self.assertEqual(list(misc.char_range("a", "d", 2)), ["a", "c"])
        self.assertEqual(list(misc.char_range("e", "a")), [])


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
        inv_ar = misc.tensorinv_numpy_array(ar, a_axes=(0, 3), b_axes=(1, 4))
        res_id = misc.tensormul_numpy_array(
            ar, inv_ar,
            a_axes=(0, 1, 2, 3, 4), b_axes=(1, 5, 2, 4, 6),
            res_axes=(0, 5, 2, 3, 6)
        )
        self.assertTrue(np.all(np.isclose(res_id, 0, atol=1e-5)
                               | np.isclose(res_id, 1)))

    def test_tensorsolve(self):
        ar = np.sqrt(np.arange(2 * 3 * 4 * 2 * 3).reshape((2, 3, 4, 2, 3)))
        y = np.arange(2 * 3 * 4).reshape((2, 3, 4))
        x_slv = misc.tensorsolve_numpy_array(
            ar, y, a_axes=(0, 1), b_axes=(3, 4), res_axes=(0, 1)
        )
        y_slv = misc.tensormul_numpy_array(
            ar, x_slv,
            a_axes=(0, 1, 2, 3, 4), b_axes=(3, 4, 2),
            res_axes=(0, 1, 2)
        )
        self.assertTrue(np.allclose(y_slv, y))


###############################################################################


if __name__ == '__main__':
    ArrayFunctionsTestCase().test_tensorsolve()
    unittest.main()
