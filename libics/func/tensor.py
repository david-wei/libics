import numpy as np

from libics.util import misc


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


def tensorinv_numpy_array(ar, a_axes=-1, b_axes=-2):
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


def tensorsolve_numpy_array(ar, res, a_axes=-1, b_axes=-2, res_axes=-1):
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
