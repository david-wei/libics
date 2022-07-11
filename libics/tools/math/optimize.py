import numpy as np


###############################################################################


def minimize_discrete_stepwise(
    fun, x0, args=(), dx=1, bounds=None,
    maxiter=10000, results_cache=None
):
    """
    Minimizes a discrete function by nearest neighbour descent.

    Parameters
    ----------
    fun : `callable`
        Function to be minimized.
        Its signature must be `fun(x0, *args) -> float`.
    x0 : `Array[1]` or `Scalar`
        Initial guess for solution.
    args : `tuple(Any)`
        Additional function arguments.
    dx : `Array[1]` or `Scalar`
        Discrete steps along each dimension.
        If scalar, applies given step to all dimensions.
    maxiter : `int`
        Maximum number of optimization steps.
    results_cache : `dict` or `None`
        Dictionary of pre-calculated results.

    Returns
    -------
    x : `Array[1, float]` or `float`
        Solution. Scalar or vectorial depending on `x0`.
    """
    # Parse parameters
    if results_cache is None:
        results_cache = {}
    else:
        results_cache = results_cache.copy()
    is_scalar = np.isscalar(x0) and np.isscalar(dx)
    if is_scalar:
        x0, dx = [x0], [dx]
    else:
        if np.isscalar(x0):
            x0 = np.full_like(dx, x0)
        if np.isscalar(dx):
            dx = np.full_like(x0, dx)
    if bounds is not None:
        raise NotImplementedError("`bounds` not yet implemented")
    # Initialize optimization variables
    x, dx = np.array(x0), np.array(dx)               # [ndim]
    dx_ar = dx[:, np.newaxis] * np.arange(-1, 2)     # [ndim, 3]
    dx_ar_mg = np.meshgrid(*dx_ar, indexing="ij")    # [ndim, {nsteps}]
    dx_ar_mg = [np.ravel(_dx) for _dx in dx_ar_mg]   # [ndim, nsteps]
    dx_ar_mg = np.transpose(dx_ar_mg)                # [nsteps, ndim]
    size = dx_ar_mg.shape[0]
    # Perform optimization
    converged = False
    for _ in range(maxiter):
        # Get result for each step direction
        res_mg = np.full(size, np.nan, dtype=float)  # [nsteps]
        x_mg = dx_ar_mg + x                          # [nsteps, ndim]
        for idx in range(size):
            key = tuple(x_mg[idx])
            if key not in results_cache:
                results_cache[key] = fun(x_mg[idx], *args)
            res_mg[idx] = results_cache[key]
        # Find best step direction
        idx_min = np.argmin(res_mg)
        x = x_mg[idx_min]
        # Check convergence
        if np.allclose(dx_ar_mg[idx_min], 0):
            converged = True
            break
    # Return results
    if not converged:
        raise RuntimeError(
            "`maxiter` reached without convergence "
            f"(`x` = {str(x)})"
        )
    if is_scalar:
        return x[0]
    else:
        return x
