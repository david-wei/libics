import numpy as np

from libics.util import misc


###############################################################################
# Tensor Operations
###############################################################################


def vectorize_numpy_array(
    ar, tensor_axes=(-2, -1), vec_axis=-1, ret_shape=False
):
    """
    Vectorizes a tensor (high-dimensional) `numpy` array. Opposite of
    :py:func:`.tensorize_numpy_array`.

    Parameters
    ----------
    ar : `np.ndarray`
        Tensor array.
    tensor_axes : `tuple(int)`
        Tensor axes to be vectorized. Vectorization is performed
        in C-like order.
    vec_axis : `int`
        Vectorized dimension of resulting vector array.
    ret_shape : `bool`
        Flag whether to return the shape of the vectorized dimensions.

    Returns
    -------
    vec : `np.ndarray`
        Vector array.
    vec_shape : `tuple(int)`
        If ret_shape is set: shape of vectorized dimensions.
        Useful as parameter for tensorization.

    Notes
    -----
    Performance is maximal for back-aligned, ordered vectorization since
    in-place data ordering is possible.

    Examples
    --------
    Given a tensor A[i, j, k, l], required is a vectorized version
    A[i, j*k, l]. The corresponding call would be:

    >>> i, j, k, l = 2, 3, 4, 5
    >>> A = np.arange(i * j * k * l).reshape((i, j, k, l))
    >>> A.shape
    (2, 3, 4, 5)
    >>> B = vectorize_numpy_array(A, tensor_axes=(1, 2), vec_axis=1)
    >>> B.shape
    (2, 12, 5)
    """
    tensor_axes = misc.assume_iter(tensor_axes)
    vec_dims = len(tensor_axes)
    vec = np.moveaxis(
        ar, tensor_axes, np.arange(-vec_dims, 0)
    )
    vec = vec.reshape(vec.shape[:-vec_dims] + (-1, ))
    vec = np.moveaxis(vec, -1, vec_axis)
    if ret_shape:
        vec_shape = tuple(np.array(ar.shape)[np.array(tensor_axes)])
        return vec, vec_shape
    else:
        return vec


def tensorize_numpy_array(
    vec, tensor_shape, tensor_axes=(-2, -1), vec_axis=-1
):
    """
    Tensorizes a vectorized `numpy` array. Opposite of
    :py:func:`.vectorize_numpy_array`.

    Parameters
    ----------
    vec : `np.ndarray`
        Vector array.
    tensor_shape : `tuple(int)`
        Shape of tensorized dimensions.
    tensor_axes : `tuple(int)`
        Tensor axes of resulting array. Tensorization is performed
        in C-like order.
    vec_axis : `int`
        Dimension of vector array to be tensorized.

    Returns
    -------
    ar : `np.ndarray`
        Tensor array.
    """
    tensor_shape = misc.assume_iter(tensor_shape)
    tensor_axes = misc.assume_iter(tensor_axes)
    vec_dims = len(tensor_axes)
    if vec_dims != len(tensor_shape):
        raise ValueError("incommensurate tensor dimensions ({:s}, {:s}"
                         .format(str(tensor_shape), str(tensor_axes)))
    ar = np.moveaxis(vec, vec_axis, -1)
    ar = ar.reshape(ar.shape[:-1] + tensor_shape)
    ar = np.moveaxis(ar, np.arange(-vec_dims, 0), tensor_axes)
    return ar


def _generate_einstr(idx):
    idx = misc.assume_iter(idx)
    ein_ls = []
    offset = ord('i')
    for i in tuple(idx):
        ein_ls.append(
            "..." if i == ... else chr(offset + i)
        )
    einstr = "".join(ein_ls)
    return einstr


def tensormul_numpy_array(
    a, b, a_axes=(..., 0), b_axes=(0, ...), res_axes=(..., )
):
    """
    Einstein-sums two `numpy` tensors.

    Parameters
    ----------
    a, b : `np.ndarray`
        Tensor operands.
    a_axes, b_axes, res_axes : `tuple(int)`
        Indices of Einstein summation. Dimensions are interpreted
        relative. Allows for use of ellipses.

    Returns
    -------
    res : `np.ndarray`
        Combined result tensor.

    Notes
    --------
    This function wraps `numpy.einsum`.

    Examples
    --------
    Denote: (a_axes), (b_axes) -> (res_axes) [`np.einsum` string]

    * Matrix multiplication: e.g.
      (0, 1), (1, 2) -> (0, 2) ["ij,jk->ik"].
    * Tensor dot: e.g.
      (0, 1, 2, 3), (4, 2, 3, 5) -> (0, 1, 4, 5) ["ijkl,mjkn->ilmn"].
    * Tensor dot with broadcasting: e.g.
      (0, 1, 2, 3), (0, 1, 2, 3) -> (0, 3) ["ijkl,ijkl->il"].
    """
    einstr = (_generate_einstr(a_axes) + "," + _generate_einstr(b_axes)
              + "->" + _generate_einstr(res_axes))
    return np.einsum(einstr, a, b)


def _matricize_numpy_array(ar, a_axes, b_axes):
    """
    Reduces a high-dimensional tensor to a right-aligned matrix.
    This matrix has the shape `(np.prod(a_axes), np.prod(b_axes))`.
    The left-aligned dimensions are broadcastable dimensions.
    Opposite of :py:func:`._unmatricize_numpy_array`.
    """
    a_axes = np.array(a_axes, ndmin=1) % ar.ndim
    b_axes = np.array(b_axes, ndmin=1) % ar.ndim
    a_vecaxes = []
    for i, a in enumerate(a_axes):
        _tmp = 0
        for b in b_axes:
            if b < a:
                _tmp += 1
        a_vecaxes.append(a - _tmp)

    vec, b_shape = vectorize_numpy_array(
        ar, tensor_axes=b_axes, vec_axis=-1, ret_shape=True
    )
    vec, a_shape = vectorize_numpy_array(
        vec, tensor_axes=a_vecaxes, vec_axis=-2, ret_shape=True
    )
    return vec, a_shape, b_shape, a_vecaxes, b_axes


def _unmatricize_numpy_array(vec, a_shape, b_shape, a_vecaxes, b_axes):
    """
    Reconstructs a right-aligned matrix to a high-dimensional tensor.
    Performs the inverse operation to :py:func:`_matricize_numpy_array`.
    """
    ar = tensorize_numpy_array(vec, a_shape,
                               tensor_axes=a_vecaxes, vec_axis=-2)
    ar = tensorize_numpy_array(ar, b_shape, tensor_axes=b_axes, vec_axis=-1)
    return ar


def tensortranspose_numpy_array(ar, a_axes=-2, b_axes=-1):
    """
    Transposes the matrix spanned by `a_axes`, `b_axes`.

    Parameters
    ----------
    ar : `np.ndarray`
        Full tensor.
    a_axes, b_axes : `tuple(int)`
        Corresponding dimensions to be transposed.
        These specified dimensions span an effective square matrix.

    Returns
    -------
    ar : `np.ndarray`
        Transposed full tensor.
    """
    a_axes, b_axes = misc.assume_iter(a_axes), misc.assume_iter(b_axes)
    for i, a in enumerate(a_axes):
        ar = np.swapaxes(ar, a, b_axes[i])
    return ar


def tensorinv_numpy_array(ar, a_axes=-2, b_axes=-1):
    """
    Calculates the tensor inverse (w.r.t. to :py:func:`tensormul_numpy_array`).

    Parameters
    ----------
    ar : `np.ndarray`
        Full tensor.
    a_axes, b_axes : `tuple(int)`
        Corresponding dimensions to be inverted.
        These specified dimensions span an effective square matrix.

    Returns
    -------
    ar : `np.ndarray`
        Inverted full tensor.
    """
    vec, a_shape, b_shape, a_vecaxes, b_axes = _matricize_numpy_array(
        ar, a_axes, b_axes
    )
    inv_vec = np.linalg.inv(vec)
    ar = _unmatricize_numpy_array(
        inv_vec, a_shape, b_shape, a_vecaxes, b_axes
    )
    return ar


def tensorsolve_numpy_array(ar, res, a_axes=-2, b_axes=-1, res_axes=-1):
    """
    Solves a tensor equation :math:`A x = b` for :math:`x`, where all operands
    may be high-dimensional.

    Parameters
    ----------
    ar : `np.ndarray`
        Matrix tensor :math:`A`.
    res : `np.ndarray`
        Vector tensor :math:`b`.
    a_axes, b_axes, res_axes : `tuple(int)`
        Tensorial indices corresponding to
        :math:`\\sum_{b} A_{ab} x_b = y_{res}`.

    Returns
    -------
    sol : `np.ndarray`
        Solution vector tensor :math:`x` with solution indices res_axes.
    """
    ar_vec, a_shape, b_shape, a_vecaxes, b_axes = _matricize_numpy_array(
        ar, a_axes, b_axes
    )
    res_vec, res_shape = vectorize_numpy_array(
        res, tensor_axes=res_axes, vec_axis=-1, ret_shape=True
    )
    sol_vec = np.linalg.solve(ar_vec, res_vec)
    sol = tensorize_numpy_array(
        sol_vec, res_shape, tensor_axes=res_axes, vec_axis=-1
    )
    return sol


def complex_norm(ar, vec_axes=None):
    r"""
    Computes the pseudonorm :math:`\sqrt{x^T x}` on the complex (tensorial)
    vector :math:`x`.

    Parameters
    ----------
    ar : `np.ndarray`
        Array constituting the vector to be normalized.
    vec_axes : `tuple(int)`
        Tensorial indices corresponding to the vectorial dimensions.

    Returns
    -------
    norm : `np.ndarray`
        Resulting norm with removed vectorial axes.
    """
    ar = misc.assume_numpy_array(ar)
    if vec_axes is None:
        vec_axes = tuple(range(ar.ndim))
    vec_axes = misc.assume_tuple(vec_axes)
    vec = vectorize_numpy_array(ar, tensor_axes=vec_axes, vec_axis=-1)
    norm = np.einsum("...i,...i", vec, vec)
    return norm


###############################################################################
# Eigensystems
###############################################################################


class LinearSystem(object):

    r"""
    Linear system solver for arbitrary complex matrices.

    The linear system is defined as

    .. math::
        M x = y,

    where :math:`M` is the matrix, :math:`x` is the solution vector, and
    :math:`y` is the result vector.

    For a given matrix :math:`M` and solution vector :math:`x`, this class
    supports evaluation of :math:`y`. If :math:`y` is given, the system
    supports direct solving for :math:`x`. The matrix and vector dimensions
    can consist of multiple indices which are automatically linearized.
    For an eigensystem solver, refer to :py:class:`DiagonalizableLS`.

    Parameters
    ----------
    matrix : `np.ndarray`
        Possibly high rank tensor representable as diagonalizable
        matrix, defining a linear system. There must be two sets of
        dimensions which can be reshaped as square matrix. The remaining
        dimensions will be broadcasted.
    mata_axes, matb_axes, vec_axes : `tuple(int)`
        Axes representing the multidimensional indices of the matrix.
        :math:`M = M_{\{a_1, ...\}, \{b_1, ...\}}, x, y = x, y_{\{v_1, ...\}}`.
        :math:`M x = y = \sum_b M_{a, b} x_b = y_a`.
        The shape of each dimension must be identical and the total number
        defines the degrees of freedom :math:`n_{\text{dof}} = \prod_i n_i`.
    """

    def __init__(
        self, matrix=None, mata_axes=-2, matb_axes=-1, vec_axes=-1
    ):
        # Tensor dimensions
        self._mata_axes = None  # Tensor dimensions for mat dim i (ij, j -> i)
        self._matb_axes = None  # Tensor dimensions for mat dim j (ij, j -> i)
        self._vec_axes = None   # Tensor dimensions for vector
        # Linear system variables
        self._factor = None     # Factor to normalize matrix
        self._matrix = None     # [..., n_dof, n_dof]
        self._result = None     # [..., n_dof]
        self._solution = None   # [..., n_dof]
        # Internal temporary variables
        self._TMP_shape = None
        self._TMP_a_vecaxes = None
        self._TMP_b_axes = None
        self._TMP_vec_axis = -1
        # Assign init args
        self.mata_axes = mata_axes
        self.matb_axes = matb_axes
        self.vec_axes = vec_axes
        self.matrix = matrix

    @property
    def mata_axes(self):
        return self._mata_axes

    @mata_axes.setter
    def mata_axes(self, val):
        if val is not None:
            self._mata_axes = misc.assume_tuple(val)

    @property
    def matb_axes(self):
        return self._matb_axes

    @matb_axes.setter
    def matb_axes(self, val):
        if val is not None:
            self._matb_axes = misc.assume_tuple(val)

    @property
    def vec_axes(self):
        return self._vec_axes

    @vec_axes.setter
    def vec_axes(self, val):
        if val is not None:
            self._vec_axes = misc.assume_tuple(val)

    @property
    def matrix(self):
        return self._factor * self._unmatricize(self._matrix)

    @matrix.setter
    def matrix(self, val):
        if val is not None:
            val = misc.assume_numpy_array(val)
            self._factor = np.abs(np.max(val))
            self._matrix = self._matricize(val / self._factor)

    @property
    def result(self):
        return self._factor * self._unvectorize(self._result)

    @result.setter
    def result(self, val):
        val = misc.assume_numpy_array(val)
        self._result = self._vectorize(val) / self._factor

    @property
    def solution(self):
        return self._unvectorize(self._solution)

    @solution.setter
    def solution(self, val):
        val = misc.assume_numpy_array(val)
        self._solution = self._vectorize(val)

    # +++++++++++++++++++++++++++++++++++++++++++

    def _matricize(self, ar):
        (
            mat, self._TMP_shape, self._TMP_shape,
            self._TMP_a_vecaxes, self._TMP_b_axes
        ) = _matricize_numpy_array(
            ar, self._mata_axes, self._matb_axes
        )
        return mat

    def _unmatricize(self, mat):
        ar = _unmatricize_numpy_array(
            mat, self._TMP_shape, self._TMP_shape,
            self._TMP_a_vecaxes, self._TMP_b_axes
        )
        return ar

    def _vectorize(self, ar):
        vec, self._TMP_shape = vectorize_numpy_array(
            ar, tensor_axes=self._vec_axes, vec_axis=self._TMP_vec_axis,
            ret_shape=True
        )
        return vec

    def _unvectorize(self, vec):
        ar = tensorize_numpy_array(
            vec, self._TMP_shape,
            tensor_axes=self._vec_axes, vec_axis=self._TMP_vec_axis
        )
        return ar

    # +++++++++++++++++++++++++++++++++++++++++++

    def solve(self, res_vec=None):
        """
        For a given result vector :math:`y`, directly solves for the solution
        vector :math:`x`.
        """
        if res_vec is None:
            res_vec = self._result
        else:
            res_vec = self._vectorize(res_vec)
        # TODO: scalable solution for non-square matrices
        self._solution = tensorsolve_numpy_array(
            self._matrix, res_vec, a_axes=-2, b_axes=-1, res_axes=-1
        )

    def eval(self, sol_vec=None):
        """
        For a given solution vector :math:`x`, directly evaluates the result
        vector :math:`y`.
        """
        if sol_vec is None:
            sol_vec = self._solution
        else:
            sol_vec = self._vectorize(sol_vec)
        self._result = tensormul_numpy_array(
            self._matrix, sol_vec,
            a_axes=(..., 0, 1), b_axes=(..., 1), res_axes=(..., 0)
        )


class DiagonalizableLS(LinearSystem):

    r"""
    Eigensystem solver for arbitrary square diagonalizable matrices.

    The linear system is defined as

    .. math::
        M x = y, x = \sum_p b_p m_p,

    where :math:`M` is the matrix, :math:`x` is the solution vector,
    :math:`y` is the result vector, :math:`(\mu_p, m_p)` is the eigensystem
    and :math:`b_p` is the eigenvector decomposition.

    If the matrix has additional properties, please use the subclasses
    :py:class:`HermitianLS` and :py:class:`SymmetricLS`.

    Notes
    -----
    Usage:

    * Initialization: Set the matrix defining the linear system. Set the axes
      for higher rank tensors.
    * Given the matrix :math:`M`, this class allows for (1.) the calculation
      of the eigensystem :math:`(\mu_p, m_p)`, (2.) solving for :math:`x`,
      and (3.) calculating the result :math:`y`.
    1. Run :py:meth:`calc_eigensystem` to calculate eigenvalues and
       left/right eigenvectors.
    2. Given the result vector :math:`y`, there are two options to solve for
       :math:`x`.

       a. If the eigensystem was calculated before, one can use
          :py:meth:`decomp_result` to obtain the eigenvector decomposition.
          Subsequently calling :py:meth:`calc_solution` calculates the
          solution vector :math:`x`.
       b. Alternatively, the solution can be obtained without eigensystem
          decomposition with :py:meth:`solve`, which only populates
          the solution vector :math:`x`.

    3. Given the solution vector :math:`x`, there are two options to obtain
       the result :math:`y`.

       a. If the eigensystem was calculated before, one can use
          :py:meth:`decomp_solution` to obtain the eigenvector
          decomposition. Subsequently calling :py:meth:`calc_result`
          calculates the result vector :math:`y`.
       b. Alternatively, the result can be obtained without eigensystem
          decomposition with :py:meth:`eval`, which only populates
          the result vector :math:`y`.
    """

    def __init__(
        self, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        # Eigensystem variables
        self._eigvals = None    # [..., n_dof]
        self._reigvecs = None   # [..., n_dof, n_components]
        self._leigvecs = None   # [..., n_dof, n_components]
        self._decomp = None     # [..., n_dof]

    @property
    def eigvals(self):
        return self._factor * self._vectorize(self._eigvals)

    @eigvals.setter
    def eigvals(self, val):
        val = misc.assume_numpy_array(val)
        self._eigvals = self._unvectorize(val / self._factor)

    @property
    def reigvecs(self):
        return self._unvectorize(self._reigvecs)

    @reigvecs.setter
    def reigvecs(self, val):
        val = misc.assume_numpy_array(val)
        self._reigvecs = self._vectorize(val)
        self._reigvecs /= np.linalg.norm(
            self._reigvecs, axis=self._TMP_vec_axis
        )

    @property
    def leigvecs(self):
        return self._unvectorize(self._leigvecs)

    @leigvecs.setter
    def leigvecs(self, val):
        val = misc.assume_numpy_array(val)
        self._leigvecs = self._vectorize(val)
        self._leigvecs /= np.linalg.norm(
            self._leigvecs, axis=self._TMP_vec_axis
        )

    @property
    def eigvecs(self):
        return self.reigvecs

    @eigvecs.setter
    def eigvecs(self, val):
        self.reigvecs(val)

    @property
    def decomp(self):
        return self._decomp

    @decomp.setter
    def decomp(self, val):
        val = misc.assume_numpy_array(val)
        self._decomp = val

    # +++++++++++++++++++++++++++++++++++++++++++

    def _calc_leigvecs(self):
        """
        Calculates the left eigenvectors from the right eigenvectors.

        Notes
        -----
        Inverts the right eigenvector matrix. To improve performance:
        overload this function if matrix has additional symmetries.
        """
        self._leigvecs = tensorinv_numpy_array(
            tensortranspose_numpy_array(self._reigvecs, a_axes=-2, b_axes=-1),
            a_axes=-2, b_axes=-1
        )

    def calc_eigensystem(self):
        """
        Calculates eigenvalues, left and right eigenvectors.
        By default the eigenvalues are sorted in a descending order.
        """
        eigvals, reigvecs = np.linalg.eig(self._matrix)
        self._eigvals = eigvals
        self._reigvecs = tensortranspose_numpy_array(
            reigvecs, a_axes=-2, b_axes=-1
        )
        self._calc_leigvecs()

    def sort_eigensystem(self, order=None):
        """
        Sorts the eigensystem according to the given order.

        Parameters
        ----------
        order : `np.ndarray(1)` or `None`
            `np.ndarray(1)`: index order defined by this array.
            `None`: index order ascending in eigenvalue.
        """
        if order is None:
            order = np.argsort(self._eigvals, axis=-1)
        self._eigvals = self._eigvals[..., order]
        self._reigvecs = self._reigvecs[..., order, :]
        self._leigvecs = self._leigvecs[..., order, :]

    def decomp_solution(self, sol_vec=None):
        """
        Decomposes a solution vector :math:`x` into an overlap vector
        :math:`b`.
        """
        if sol_vec is None:
            sol_vec = self._solution
        else:
            sol_vec = self._vectorize(sol_vec)
        self._decomp = tensormul_numpy_array(
            self._leigvecs, sol_vec,
            a_axes=(..., 0, 1), b_axes=(..., 1), res_axes=(..., 0)
        )

    def decomp_result(self, res_vec=None):
        """
        Decomposes a result vector :math:`y` into an overlap vector :math:`b`.
        """
        if res_vec is None:
            res_vec = self._result
        else:
            res_vec = self._vectorize(res_vec)
        self._decomp = tensormul_numpy_array(
            self._leigvecs / self._eigvals[..., np.newaxis], res_vec,
            a_axes=(..., 0, 1), b_axes=(..., 1), res_axes=(..., 0)
        )

    def calc_solution(self, decomp_vec=None):
        """
        Calculates the solution vector :math:`x` from a decomposition vector
        :math:`b`.
        """
        if decomp_vec is None:
            decomp_vec = self._decomp
        else:
            decomp_vec = self._vectorize(decomp_vec)
        self._solution = tensormul_numpy_array(
            decomp_vec, self._reigvecs,
            a_axes=(..., 0), b_axes=(..., 0, 1), res_axes=(..., 1)
        )

    def calc_result(self, decomp_vec=None):
        """
        Calculates the result vector :math:`y` from a decomposition vector
        :math:`b`.
        """
        if decomp_vec is None:
            decomp_vec = self._decomp
        else:
            decomp_vec = self._vectorize(decomp_vec)
        self._result = tensormul_numpy_array(
            self._eigvals * decomp_vec, self._reigvecs,
            a_axes=(..., 0), b_axes=(..., 0, 1), res_axes=(..., 1)
        )


class HermitianLS(DiagonalizableLS):

    """
    Eigensystem solver for Hermitian diagonalizable matrices.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _calc_leigvecs(self):
        self._leigvecs = np.conjugate(self._reigvecs)


class SymmetricLS(DiagonalizableLS):

    """
    Eigensystem solver for complex symmetric diagonalizable matrices.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _calc_leigvecs(self):
        cnorm = complex_norm(self._reigvecs, vec_axes=-1)
        self._reigvecs /= cnorm[..., np.newaxis]
        self._leigvecs = np.array(self._reigvecs)
