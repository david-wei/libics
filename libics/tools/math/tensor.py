import numpy as np

from libics.core.util import misc


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
        `res_axes` allows for `None` to obtain a scalar.

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
    einstr = (_generate_einstr(a_axes) + "," + _generate_einstr(b_axes))
    if res_axes is not None and len(misc.assume_tuple(res_axes)) > 0:
        einstr += "->" + _generate_einstr(res_axes)
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


def tensorsolve_numpy_array(
    ar, res, a_axes=-2, b_axes=-1, res_axes=-1, algorithm=None
):
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
    algorithm : `None` or `str`
        `None` or `"lu_fac"`: LU factorization.
        `"lst_sq"`: least squares optimization.

    Returns
    -------
    sol : `np.ndarray`
        Solution vector tensor :math:`x` with solution indices res_axes.

    Notes
    -----
    Tries to solve the linear equation using a deterministic full-rank solver.
    If this fails, a least-squares algorithm is used. The least-squares solver
    does not support broadcasting.
    """
    ar_vec, a_shape, b_shape, a_vecaxes, b_axes = _matricize_numpy_array(
        ar, a_axes, b_axes
    )
    res_vec, res_shape = vectorize_numpy_array(
        res, tensor_axes=res_axes, vec_axis=-1, ret_shape=True
    )
    try:
        if algorithm is None or algorithm == "lu_fac":
            sol_vec = np.linalg.solve(ar_vec, res_vec)
        elif algorithm == "lst_sq":
            sol_vec = np.linalg.lstsq(ar_vec, res_vec, rcond=None)[0]
        else:
            raise KeyError("invalid algorithm ({:s})".format(str(algorithm)))
    # Fallback solver using least squares optimization
    except np.linalg.LinAlgError:
        if algorithm == "lst_sq":
            raise
        else:
            sol_vec = np.linalg.lstsq(ar_vec, res_vec, rcond=None)[0]
    sol = tensorize_numpy_array(
        sol_vec, res_shape, tensor_axes=res_axes, vec_axis=-1
    )
    return sol


def euclid_norm(ar, axis=None):
    r"""
    Computes the Euclidean norm :math:`\sqrt{x^\dagger x}` on the complex
    (tensorial) vector :math:`x`.

    Parameters
    ----------
    ar : `np.ndarray`
        Array constituting the vector to be normalized.
    axis : `tuple(int)`
        Tensorial indices corresponding to the vectorial dimensions.

    Returns
    -------
    norm : `np.ndarray`
        Resulting norm with removed vectorial axes.
    """
    ar = misc.assume_numpy_array(ar)
    if axis is None:
        axis = tuple(range(ar.ndim))
    axis = misc.assume_tuple(axis)
    vec = vectorize_numpy_array(ar, tensor_axes=axis, vec_axis=-1)
    return np.linalg.norm(vec, axis=-1)


def complex_norm(ar, axis=None):
    r"""
    Computes the pseudonorm :math:`\sqrt{x^T x}` on the complex (tensorial)
    vector :math:`x`.

    Parameters
    ----------
    ar : `np.ndarray`
        Array constituting the vector to be normalized.
    axis : `tuple(int)`
        Tensorial indices corresponding to the vectorial dimensions.

    Returns
    -------
    norm : `np.ndarray`
        Resulting norm with removed vectorial axes.
    """
    ar = misc.assume_numpy_array(ar)
    if axis is None:
        axis = tuple(range(ar.ndim))
    axis = misc.assume_tuple(axis)
    vec = vectorize_numpy_array(ar, tensor_axes=axis, vec_axis=-1)
    norm = np.einsum("...i,...i", vec, vec)
    return np.sqrt(norm)


def ortho_gram_schmidt(vectors, norm_func=np.linalg.norm):
    """
    Naive Gram-Schmidt orthogonalization.

    Parameters
    ----------
    vectors : `np.ndarray`
        Normalized, linearly independent row vectors to be orthonormalized.
        Dimensions: [n_vector, n_components].
    norm : `callable`
        Function computing a vector norm.
        Call signature: `norm_func(vector)->scalar`.

    Returns
    -------
    basis : `np.ndarray`
        Orthonormal basis of space spanned by `vectors`.
        Dimensions: [n_vector, n_components].
    """
    basis = np.zeros_like(vectors)
    basis[0] = vectors[0]
    counter = 1
    for i, vec in enumerate(vectors[1:]):
        v_proj = vec - (basis[:counter] @ vec) @ basis[:counter]
        basis[i + 1] = v_proj / norm_func(v_proj)
        counter += 1
    return basis


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
    mata_axes, matb_axes, veca_axes, veca_bxes : `tuple(int)`
        Axes representing the multidimensional indices of the matrix.
        :math:`M = M_{\{a_1, ...\}, \{b_1, ...\}}, x, y = x, y_{\{v_1, ...\}}`.
        :math:`M x = y = \sum_b M_{a, b} x_b = y_a`.
        The shape of each dimension must be identical and the total number
        defines the degrees of freedom :math:`n_{\text{dof}} = \prod_i n_i`.
    """

    def __init__(
        self, matrix=None,
        mata_axes=-2, matb_axes=-1, veca_axes=-1, vecb_axes=-1, vec_axes=None
    ):
        # Tensor dimensions
        self._mata_axes = None  # Tensor dimensions for mat dim i (ij, j -> i)
        self._matb_axes = None  # Tensor dimensions for mat dim j (ij, j -> i)
        self._veca_axes = None  # Tensor dimensions for result vector
        self._vecb_axes = None  # Tensor dimensions for solution vector
        # Linear system variables
        self._matrix = None     # [..., n_dof, n_dof]
        self._result = None     # [..., n_dof]
        self._solution = None   # [..., n_dof]
        # Internal temporary variables
        self._TMP_a_shape = None
        self._TMP_b_shape = None
        self._TMP_a_vecaxes = None
        self._TMP_b_axes = None
        self._TMP_vec_axis = -1
        # Assign init args
        self.mata_axes = mata_axes
        self.matb_axes = matb_axes
        if vec_axes is None:
            self.veca_axes = veca_axes
            self.vecb_axes = vecb_axes
        else:
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
    def veca_axes(self):
        return self._veca_axes

    @veca_axes.setter
    def veca_axes(self, val):
        if val is not None:
            self._veca_axes = misc.assume_tuple(val)

    @property
    def vecb_axes(self):
        return self._vecb_axes

    @vecb_axes.setter
    def vecb_axes(self, val):
        if val is not None:
            self._vecb_axes = misc.assume_tuple(val)

    @property
    def vec_axes(self):
        if self.veca_axes == self.vecb_axes:
            return self.vecb_axes
        else:
            raise ValueError("unequal vec_axes (a/b)")

    @vec_axes.setter
    def vec_axes(self, val):
        if val is not None:
            self.veca_axes = val
            self.vecb_axes = val

    @property
    def matrix(self):
        return self._unmatricize(self._matrix)

    @matrix.setter
    def matrix(self, val):
        if val is not None:
            val = misc.assume_numpy_array(val)
            self._matrix = self._matricize(val)

    @property
    def result(self):
        return self._unvectorize_a(self._result)

    @result.setter
    def result(self, val):
        val = misc.assume_numpy_array(val)
        self._result = self._vectorize_a(val)

    @property
    def solution(self):
        return self._unvectorize_b(self._solution)

    @solution.setter
    def solution(self, val):
        val = misc.assume_numpy_array(val)
        self._solution = self._vectorize_b(val)

    # +++++++++++++++++++++++++++++++++++++++++++

    def _matricize(self, ar):
        (
            mat, self._TMP_a_shape, self._TMP_b_shape,
            self._TMP_a_vecaxes, self._TMP_b_axes
        ) = _matricize_numpy_array(
            ar, self._mata_axes, self._matb_axes
        )
        return mat

    def _unmatricize(self, mat):
        ar = _unmatricize_numpy_array(
            mat, self._TMP_a_shape, self._TMP_b_shape,
            self._TMP_a_vecaxes, self._TMP_b_axes
        )
        return ar

    def _vectorize_a(self, ar):
        vec, self._TMP_a_shape = vectorize_numpy_array(
            ar, tensor_axes=self._veca_axes, vec_axis=self._TMP_vec_axis,
            ret_shape=True
        )
        return vec

    def _unvectorize_a(self, vec):
        ar = tensorize_numpy_array(
            vec, self._TMP_a_shape,
            tensor_axes=self._veca_axes, vec_axis=self._TMP_vec_axis
        )
        return ar

    def _vectorize_b(self, ar):
        vec, self._TMP_b_shape = vectorize_numpy_array(
            ar, tensor_axes=self._vecb_axes, vec_axis=self._TMP_vec_axis,
            ret_shape=True
        )
        return vec

    def _unvectorize_b(self, vec):
        ar = tensorize_numpy_array(
            vec, self._TMP_b_shape,
            tensor_axes=self._vecb_axes, vec_axis=self._TMP_vec_axis
        )
        return ar

    def _vectorize(self, ar):
        return self._vectorize_a(ar)

    def _unvectorize(self, vec):
        return self._unvectorize_a(vec)

    def _norm(self, ar, axis=None):
        return np.linalg.norm(ar, axis=axis)

    def _get_broadcast_dim_iter(self):
        non_mat_shape = self._matrix.shape[:-2]
        if len(non_mat_shape) < 1:
            return None
        else:
            non_mat_idx = [range(i) for i in non_mat_shape]
            return misc.get_combinations(non_mat_idx)

    # +++++++++++++++++++++++++++++++++++++++++++

    def solve(self, res_vec=None, algorithm=None):
        """
        For a given result vector :math:`y`, directly solves for the solution
        vector :math:`x`.
        """
        if res_vec is None:
            res_vec = self._result
        else:
            res_vec = self._vectorize_a(res_vec)
        # TODO: scalable solution for non-square matrices
        self._solution = tensorsolve_numpy_array(
            self._matrix, res_vec, a_axes=-2, b_axes=-1, res_axes=-1,
            algorithm=algorithm
        )

    def eval(self, sol_vec=None):
        """
        For a given solution vector :math:`x`, directly evaluates the result
        vector :math:`y`.
        """
        if sol_vec is None:
            sol_vec = self._solution
        else:
            sol_vec = self._vectorize_b(sol_vec)
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

      a. If the eigenvalues should be arranged meaningfully, they can be
         ordered using :py:meth:`sort_eigensystem`.
      b. For non-symmetric and non-Hermitian matrices or for degenerate
         eigenvalues, the eigenvectors are not orthogonal. A computationally
         expensive orthonormalization can be obtained with
         :py:meth:`ortho_eigensystem`.

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
        self._eigvals = None        # [..., n_dof]
        self._reigvecs = None       # [..., n_dof, n_components]
        self._leigvecs = None       # [..., n_dof, n_components]
        self._decomp = None         # [..., n_dof]
        self._is_invertible = None  # Flag for diagonalizability

    @LinearSystem.matrix.setter
    def matrix(self, val):
        if val is not None:
            val = misc.assume_numpy_array(val)
            self._matrix = self._matricize(val)
            self._is_invertible = None

    @property
    def eigvals(self):
        return self._unvectorize(self._eigvals)

    @eigvals.setter
    def eigvals(self, val):
        val = misc.assume_numpy_array(val)
        self._eigvals = self._vectorize(val)

    @property
    def reigvecs(self):
        return self._unmatricize(self._reigvecs)

    @reigvecs.setter
    def reigvecs(self, val):
        val = misc.assume_numpy_array(val)
        self._reigvecs = self._matricize(val)
        self._reigvecs /= np.linalg.norm(
            self._reigvecs, axis=self._TMP_vec_axis
        )

    @property
    def leigvecs(self):
        return self._unmatricize(self._leigvecs)

    @leigvecs.setter
    def leigvecs(self, val):
        val = misc.assume_numpy_array(val)
        self._leigvecs = self._matricize(val)
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
        return self._unvectorize(self._decomp)

    @decomp.setter
    def decomp(self, val):
        val = misc.assume_numpy_array(val)
        self._decomp = self._vectorize(val)

    @property
    def is_invertible(self):
        """
        Checks the rank of the matrix. Can be computationally expensive!
        """
        if self._is_invertible is None:
            a_dim, b_dim = self._matrix.shape[-2:]
            self._is_invertible = (a_dim == b_dim)
            if self._is_invertible:
                self._is_invertible = np.all(
                    np.linalg.matrix_rank(self._matrix) == a_dim
                )
        return self._is_invertible

    @property
    def is_diagonalizable(self):
        return self.is_invertible

    @property
    def is_singular(self):
        return not self.is_invertible

    @property
    def is_defective(self):
        return not self.is_invertible

    # +++++++++++++++++++++++++++++++++++++++++++

    def calc_eigensystem(self):
        """
        Calculates eigenvalues, normalized left and right eigenvectors.

        Notes
        -----
        * The eigenvalues are in no guaranteed order.
          See :py:meth:`sort_eigensystem`.
        * The eigenvectors are not necessarily orthogonal.
          See :py:meth:`ortho_eigensystem`.
        """
        eigvals, reigvecs = np.linalg.eig(self._matrix)
        self._eigvals = eigvals
        self._reigvecs = tensortranspose_numpy_array(
            reigvecs, a_axes=-2, b_axes=-1
        )
        self._calc_leigvecs()

    def _calc_leigvecs(self):
        self._leigvecs = tensorinv_numpy_array(
            tensortranspose_numpy_array(self._reigvecs, a_axes=-2, b_axes=-1),
            a_axes=-2, b_axes=-1
        )

    def sort_eigensystem(self, order=None):
        """
        Sorts the eigensystem according to the given order.

        Parameters
        ----------
        order : `np.ndarray` or `callable` or `None`
            `np.ndarray`:
                Index order defined by this array.
                Dimensions: [n_dof].
            `callable`:
                Eigenvalue measurement function whose ascendingly sorted
                return value defines the index order.
                Call signature: `func(np.ndarray(..., n_dof))->float`.
            `None`:
                Index order ascending in modulus of eigenvalue.
        """
        if order is None:
            order = np.argsort(np.abs(self._eigvals), axis=-1)
        elif callable(order):
            order = np.argsort(order(self._eigvals), axis=-1)
        elif order.ndim > 1:
            order = self._vectorize(order)
        self._eigvals = self._eigvals[..., order]
        self._reigvecs = self._reigvecs[..., order, :]
        self._leigvecs = self._leigvecs[..., order, :]

    def ortho_eigensystem(self):
        """
        Orthonormalizes the eigenvectors.
        """
        bc_indices = self._get_broadcast_dim_iter()
        if bc_indices is None:
            bc_indices = [tuple()]  # Enter the loop once
        for bc_idx in bc_indices:
            nonunique_indices = misc.extract_index_nonunique_array(
                self._eigvals[bc_idx], min_count=2
            )
            for idx in nonunique_indices:
                self._reigvecs[bc_idx + (idx, )] = ortho_gram_schmidt(
                    self._reigvecs[bc_idx + (idx, )], norm_func=self._norm
                )
        self._calc_leigvecs()

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
            decomp_vec * self._eigvals, self._reigvecs,
            a_axes=(..., 0), b_axes=(..., 0, 1), res_axes=(..., 1)
        )


class HermitianLS(DiagonalizableLS):

    """
    Eigensystem solver for Hermitian diagonalizable matrices.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def calc_eigensystem(self):
        eigvals, reigvecs = np.linalg.eig(self._matrix)
        self._eigvals = eigvals
        self._reigvecs = tensortranspose_numpy_array(
            reigvecs, a_axes=-2, b_axes=-1
        )
        self._calc_leigvecs()

    def _calc_leigvecs(self):
        self._leigvecs = np.conjugate(self._reigvecs)


class SymmetricLS(DiagonalizableLS):

    """
    Eigensystem solver for complex symmetric diagonalizable matrices.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _norm(self, ar, axis=None):
        return complex_norm(ar, axis=axis)

    def calc_eigensystem(self):
        eigvals, reigvecs = np.linalg.eig(self._matrix)
        self._eigvals = eigvals
        self._reigvecs = tensortranspose_numpy_array(
            reigvecs, a_axes=-2, b_axes=-1
        )
        self._reigvecs /= self._norm(self._reigvecs, axis=-1)[..., np.newaxis]
        self._calc_leigvecs()

    def _calc_leigvecs(self):
        self._leigvecs = self._reigvecs.copy()
