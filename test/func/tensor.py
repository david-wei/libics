import unittest
import numpy as np

from libics.func import tensor


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
        vec, vec_shape = tensor.vectorize_numpy_array(
            ar, tensor_axes=tensor_axes, vec_axis=vec_axis, ret_shape=True
        )
        res_ar = tensor.tensorize_numpy_array(
            vec, vec_shape, tensor_axes=tensor_axes, vec_axis=vec_axis
        )
        self.assertTrue(np.allclose(ar, res_ar))

    def test_tensormul_tensorinv(self):
        # Matrix and broadcasting test
        ar = np.sqrt(np.arange(2 * 3 * 3)).reshape((2, 3, 3))
        inv_ar = tensor.tensorinv_numpy_array(ar, a_axes=1, b_axes=2)
        res_id = tensor.tensormul_numpy_array(
            ar, inv_ar, a_axes=(0, 1, 2), b_axes=(0, 2, 3), res_axes=(0, 1, 3)
        )
        self.assertTrue(np.all(np.isclose(res_id, 0) | np.isclose(res_id, 1)))
        # Tensor dot and broadcasting test
        ar = np.sqrt(np.arange(2 * 2 * 2 * 3 * 3)).reshape((2, 2, 2, 3, 3))
        inv_ar = tensor.tensorinv_numpy_array(ar, a_axes=(0, 3), b_axes=(1, 4))
        res_id = tensor.tensormul_numpy_array(
            ar, inv_ar,
            a_axes=(0, 1, 2, 3, 4), b_axes=(1, 5, 2, 4, 6),
            res_axes=(0, 5, 2, 3, 6)
        )
        self.assertTrue(np.all(np.isclose(res_id, 0, atol=1e-5)
                               | np.isclose(res_id, 1)))

    def test_tensorsolve(self):
        ar = np.sqrt(np.arange(2 * 3 * 4 * 2 * 3).reshape((2, 3, 4, 2, 3)))
        y = np.arange(2 * 3 * 4).reshape((2, 3, 4))
        x_slv = tensor.tensorsolve_numpy_array(
            ar, y, a_axes=(0, 1), b_axes=(3, 4), res_axes=(0, 1)
        )
        y_slv = tensor.tensormul_numpy_array(
            ar, x_slv,
            a_axes=(0, 1, 2, 3, 4), b_axes=(3, 4, 2),
            res_axes=(0, 1, 2)
        )
        self.assertTrue(np.allclose(y_slv, y))


###############################################################################


class LinearSystemTestCase(unittest.TestCase):

    def _generate_random_array(self, shape=(1, 2, 3, 2, 3)):
        return (
            np.random.rand(*shape) + 1j * np.random.rand(*shape)
        )

    def setUp(self):
        self.mat_shape = (3, 4, 5, 4, 5)
        self.vec_shape = (3, 4, 5)
        self.mata_axes = (1, 2)
        self.matb_axes = (3, 4)
        self.vec_axes = (1, 2)
        self.mula_axes = (0, 1, 2, 3, 4)
        self.mulb_axes = (0, 3, 4)
        self.mulc_axes = (0, 3, 4, 5, 6)
        self.mulres_axes = (0, 1, 2)
        self.mulc_axes_t = (0, 5, 6, 3, 4)
        self.mulcres_axes = (0, 1, 2, 5, 6)

        np.random.seed(0)
        self.mat = self._generate_random_array(self.mat_shape)
        self.hmat = self._generate_random_array(self.mat_shape)
        self.hmat = (self.hmat + np.conjugate(
            tensor.tensortranspose_numpy_array(
                self.hmat, a_axes=self.mata_axes, b_axes=self.matb_axes
            )
        )) / 2
        self.smat = self._generate_random_array(self.mat_shape)
        self.smat = (self.smat + tensor.tensortranspose_numpy_array(
            self.smat, a_axes=self.mata_axes, b_axes=self.matb_axes
        )) / 2
        self.x = self._generate_random_array(self.vec_shape)
        self.y = tensor.tensormul_numpy_array(
            self.mat, self.x,
            a_axes=self.mula_axes, b_axes=self.mulb_axes,
            res_axes=self.mulres_axes
        )
        self.hy = tensor.tensormul_numpy_array(
            self.hmat, self.x,
            a_axes=self.mula_axes, b_axes=self.mulb_axes,
            res_axes=self.mulres_axes
        )
        self.sy = tensor.tensormul_numpy_array(
            self.smat, self.x,
            a_axes=self.mula_axes, b_axes=self.mulb_axes,
            res_axes=self.mulres_axes
        )

    def tearDown(self):
        pass

    def test_linear_system(self):
        ls = tensor.LinearSystem(
            matrix=self.mat, mata_axes=self.mata_axes,
            matb_axes=self.matb_axes, vec_axes=self.vec_axes
        )
        # Evaluate
        ls.solution = self.x.copy()
        ls.eval()
        self.assertTrue(np.allclose(ls.result, self.y))

    def test_diagonalizable_ls(self):
        ls = tensor.DiagonalizableLS(
            matrix=self.mat, mata_axes=self.mata_axes,
            matb_axes=self.matb_axes, vec_axes=self.vec_axes
        )
        # Solve
        ls.result = self.y.copy()
        ls.solve()
        self.assertTrue(np.allclose(ls.solution, self.x))
        # Eigensystem
        ls.calc_eigensystem()
        # Decompose solution
        ls.solution = self.x.copy()
        ls.decomp_solution()
        ref_decomp = ls.decomp.copy()
        ls.calc_result()
        self.assertTrue(np.allclose(ls.result, self.y))
        # Decompose result
        ls.result = self.y.copy()
        ls.decomp_result()
        self.assertTrue(np.allclose(ls.decomp, ref_decomp))
        ls.calc_solution()
        self.assertTrue(np.allclose(ls.solution, self.x))

    def test_derived_ls(self):
        # Hermitian reference
        ls = tensor.DiagonalizableLS(
            matrix=self.hmat, mata_axes=self.mata_axes,
            matb_axes=self.matb_axes, vec_axes=self.vec_axes
        )
        self.assertTrue(ls.is_invertible)
        ls.calc_eigensystem()
        ls.solution = self.x.copy()
        ls.decomp_solution()
        # Hermitian linear system
        hls = tensor.HermitianLS(
            matrix=self.hmat, mata_axes=self.mata_axes,
            matb_axes=self.matb_axes, vec_axes=self.vec_axes
        )
        hls.calc_eigensystem()
        self.assertTrue(np.allclose(hls.leigvecs, ls.leigvecs))
        hls.solution = self.x.copy()
        hls.decomp_solution()
        hls.calc_solution()
        hls.calc_result()
        self.assertTrue(np.allclose(hls.decomp, ls.decomp))
        self.assertTrue(np.allclose(hls.solution, self.x))
        self.assertTrue(np.allclose(hls.result, self.hy))
        # Complex symmetric linear system
        sls = tensor.SymmetricLS(
            matrix=self.smat, mata_axes=self.mata_axes,
            matb_axes=self.matb_axes, vec_axes=self.vec_axes
        )
        sls.calc_eigensystem()
        sls.solution = self.x.copy()
        sls.decomp_solution()
        sls.calc_solution()
        sls.calc_result()
        self.assertTrue(np.allclose(sls.solution, self.x))
        self.assertTrue(np.allclose(sls.result, self.sy))
        decomp = sls.decomp.copy()
        sls.result = self.sy.copy()
        sls.decomp_result()
        sls.calc_result()
        sls.calc_solution()
        self.assertTrue(np.allclose(sls.solution, self.x))
        self.assertTrue(np.allclose(sls.result, self.sy))
        self.assertTrue(np.allclose(sls.decomp, decomp))


###############################################################################


if __name__ == '__main__':
    np.set_printoptions(linewidth=400)
    unittest.main()
