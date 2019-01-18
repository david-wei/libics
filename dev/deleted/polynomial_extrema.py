"""
Numpy vectorized polynomial extremum finding algorithm.
"""


# Sampled polynomial evaluation and array extremum search
mask = self.get_mask()
probe_disp = np.linspace(
    self.displacements.min(), self.displacements.max(), num=50
)[np.newaxis, ...]
powers = np.arange(poly_deg + 1)[..., np.newaxis]
disp_powers = np.zeros((poly_deg + 1, probe_disp.shape[1]),
                        dtype=float)
temp_res = np.zeros(len(probe_disp), dtype=float)
params = self.scoh_params[:, mask].T[..., np.newaxis]
self.scoh_offset = np.full_like(mask, np.nan, dtype=float)
for i in range(len(mask[mask])):
    print("[{:d}/{:d}]".format(i, len(mask[mask])), end="\r")
    np.power(probe_disp, powers, out=disp_powers)
    np.multiply(params[i], disp_powers, out=disp_powers)
    np.sum(disp_powers, axis=0, out=temp_res)
    self.scoh_offset[mask][i] = probe_disp[0, temp_res.argmax()]
print("           ", end="\r")


# Numerical minimization of continuous polynomial
mask = self.get_mask()
self.scoh_offset = np.full_like(mask, np.nan, dtype=float)
params = self.scoh_params[:, mask].T
bounds = (self.displacements.min(), self.displacements.max())
for i in range(len(mask[mask])):
    print("[{:d}/{:d}]".format(i, len(mask[mask])), end="\r")
    res = optimize.minimize_scalar(
        lambda x: -np.polynomial.polynomial.polyval(x, params[i]),
        bounds=bounds, method="bounded"
    )
    self.scoh_offset[mask][i] = res.x
print("           ", end="\r")


# Perform a constrained re-fit for non-convergent fits
if poly_deg > 2:
    # Reload spatial coherence to avoid NaN interpolation
    scoh = np.array([item.spatial_coherence for item in self.items])
    coords = np.argwhere(np.logical_or(
        self.scoh_params[6] > 0, self.scoh_params[2] > 0
    ))
    sigma = 1 / weights
    bounds = np.full((poly_deg + 1, 2), np.inf)
    bounds[0] *= -1
    bounds[2, 1], bounds[6, 1] = 0, 0
    p0 = np.array([1, 1, -1, 1, 1, 1, -1], dtype=float)
    max_func_evals = 400 * len(disp + 1)

    def fit_func_poly(x, *args):
        return np.polynomial.polynomial.polyval(x, args)

    for c in coords:
        if np.any(np.isnan(scoh[:, c[0], c[1]])):
            continue
        param, cov = optimize.curve_fit(
            fit_func_poly, disp, scoh[:, c[0], c[1]],
            p0=p0, sigma=sigma, maxfev=max_func_evals
        )
        if np.inf in cov:
            print(c, param)
        self.scoh_params[:, c[0], c[1]] = param


# Calculate coherence length
mask = self.get_mask()
self.scoh_length = np.full_like(mask, np.nan, dtype=float)
scoh_params = self.scoh_params[:, mask]
scoh_scale = self.scoh_scale[mask][..., np.newaxis]
quad_order = 50
bound_left = self.displacements.min()
bound_right = self.displacements.max()
self.scoh_length[mask], _ = integrate.fixed_quad(
    lambda x: (
        np.exp(2 * np.polynomial.polynomial.polyval(x, scoh_params))
        / scoh_scale**2
    ), bound_left, bound_right, n=quad_order
)
