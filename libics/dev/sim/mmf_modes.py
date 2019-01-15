import abc
import time

import numpy as np
from scipy import constants, integrate, interpolate, optimize, special

from libics.file import hdf
from libics.util import InheritMap


###############################################################################


class COORD:

    CARTESIAN = "cartesian"
    POLAR = "polar"


@InheritMap(map_key=("libics-dev", "Fiber"))
class Fiber(abc.ABC, hdf.HDFBase):

    """
    Fiber propagation eigensystem base class.

    Parameters
    ----------
    coord : COORD
        Coordinate system.
    opt_freq : float
        Optical frequency in Hertz (Hz).
    core_radius : float
        Fiber core radius in meter (m).
    clad_radius : float
        Fiber cladding radius in meter (m).
        [Not used].
    mode_numbers : list(tuple)
        Numbers specifying modes.
    mode_propconsts : list(float)
        Propagation constants of modes.
    """

    def __init__(
        self,
        coord=COORD.CARTESIAN, opt_freq=None,
        core_radius=None, clad_radius=None,
        mode_numbers=[], mode_propconsts=[],
        pkg_name="libics-dev", cls_name="Fiber", **kwargs
    ):
        super().__init__(pkg_name=pkg_name, cls_name=cls_name, **kwargs)
        self.coord = coord
        self.opt_freq = opt_freq
        self.core_radius = core_radius
        self.clad_radius = clad_radius
        self.mode_numbers = mode_numbers
        self.mode_propconsts = mode_propconsts
        self._set_mode_number_map()

    def _hdf_init_write(self):
        self.__mode_number_map = {}

    @property
    def internal_coord(self):
        return COORD.CARTESIAN

    @property
    def numerical_aperture(self):
        return np.sqrt(self.core_refr**2 - self.clad_refr**2)

    @property
    def vacuum_wavenumber(self):
        return 2 * np.pi * self.opt_freq / constants.speed_of_light

    def trafo_coords(self, var1, var2, coord):
        """
        Transforms the given coordinates into the internally used coordinate
        system.

        Parameters
        ----------
        var1, var2 : float or np.ndarray(float)
            Positional variables corresponding to dimensions (1, 2).
        coord : COORD
            Coordinate system the var_coords are given in.

        Returns
        -------
        var1, var2 : float or np.ndarray(float)
            Coordinates in internal coordinate system.
        """
        if self.internal_coord == COORD.CARTESIAN:
            if coord == COORD.POLAR:
                var1, var2 = (
                    var1 * np.cos(var2),
                    var1 * np.sin(var2)
                )
        elif self.internal_coord == COORD.POLAR:
            if coord == COORD.CARTESIAN:
                var1, var2 = (
                    np.sqrt(var1**2 + var2**2),
                    np.arctan2(var2, var1)
                )
        return var1, var2

    def _set_mode_number_map(self):
        """
        Sets up a dictionary which maps the mode numbers to the internally
        stored mode list index.
        """
        self.__mode_number_map = {}
        for i, mn in enumerate(self.mode_numbers):
            self.__mode_number_map[tuple(mn)] = i

    @property
    def mode(self):
        """
        Gets the mode number dictionary mapping mode numbers to mode list
        indices.
        """
        if not self.__mode_number_map:
            self._set_mode_number_map()
        return self.__mode_number_map

    @property
    def mode_count(self):
        """
        Gets the number of propagation modes.
        """
        return len(self.mode_numbers)

    @property
    def mode_numbers(self):
        """
        Gets the list of mode numbers.
        """
        return self._mode_numbers

    @mode_numbers.setter
    def mode_numbers(self, val):
        """
        Sets the list of mode numbers and constructs the mode number
        dictionary.
        """
        self._mode_numbers = val
        self._set_mode_number_map()

    # ++++ Eigensystem calculations ++++

    @abc.abstractmethod
    def calc_modes(self):
        """
        Calculates the mode eigenvalues and eigenprofiles.
        """

    @abc.abstractmethod
    def calc_overlap(self, input_field, *args, algorithm="dblquad", **kwargs):
        """
        Calculates the mode overlap vector with an input beam profile.

        Parameters
        ----------
        input_field : callable
            Function representing the input light field.
            Call signature : f (var_1, var_2, coord)
                var_1, var_2 : float
                    2D spatial variables.
                coord : COORD
                    Coordinate system type.
        *args
            Parameters passed to input_field.
        algorithm : "dblquad", "hybrid", "simpson"
            Overlap integration algorithm.
        **kwargs
            Keyword arguments passed to input_field.

        Returns
        -------
        overlap : numpy.ndarray(1, float)
            Overlap vector corresponding to the fiber modes.
        """

    # ++++ Output fields ++++++++

    @abc.abstractmethod
    def __call__(
        self, var1, var2, mode_number, coord=COORD.CARTESIAN
    ):
        """
        Evaluates the complex mode profile of a given mode number.

        Parameters
        ----------
        var1, var2:
            Positional variables in dimensions (1, 2), which are
            typically (x, y) or (r, φ).
        mode_number : (int, int) or int
            tuple: mode number.
            int: internal mode list index.
        coord : COORD
            Coordinate system the positional variable are given in.

        Returns
        -------
        mode_profile : np.ndarray(float)
            Complex mode distribution.
            Shape: shape of var_coord
        """

    def vec_field(
        self, var1, var2, coord=None
    ):
        """
        Vector of output field profiles given in the internal mode list order.

        Parameters
        ----------
        var1, var2:
            Positional variables in dimensions (1, 2), which are
            typically (x, y) or (r, φ).
        coord : COORD
            Coordinate system the positional variable are given in.

        Returns
        -------
        mode_profiles : np.ndarray(float)
            Vector of complex mode distributions.
            Shape: (number of modes, shape of var_coord).
        """
        if coord is None:
            coord = self.internal_coord
        mode_profiles = np.array([
            self(var1, var2, index)
            for index in np.arange(self.mode_count)
        ])
        return mode_profiles

    def output(self, var1, var2, overlap, fiber_length, coord=COORD.POLAR):
        """
        Evaluates the complex mode profile output for a given overlap vector.

        Parameters
        ----------
        var1, var2:
            Positional variables in dimensions (1, 2), which are
            typically (x, y) or (r, φ).
        overlap : numpy.ndarray(1)
            Overlap vector corresponding to the fiber modes.
        fiber_length : float
            Propagation length in meter (m).
        coord : COORD
            Specifies given coordinate system.
            If COORD.CARTESIAN, the parameters (radius, azimuth)
            are interpreted as (x, y).

        Returns
        -------
        output : np.ndarray(float)
            Complex output field distribution.
            Shape: shape of var_coords
        """
        var1, var2 = self.trafo_coords(
            var1, var2, coord
        )
        mode_profiles = self.vec_field(var1, var2, coord=coord)
        output = np.sum(
            overlap
            * np.exp(-1j * self.mode_propconsts * fiber_length)
            * np.moveaxis(mode_profiles, 0, -1),
            axis=-1
        )
        return output

    # ++++ Correlation matrices ++++++++

    @property
    def corr_propconst(self):
        """
        Mode propagation constant correlation matrix B_mn = β_m - β_n.
        """
        corr_propconst = np.subtract.outer(
            self.mode_propconsts, self.mode_propconsts
        )
        return corr_propconst

    def corr_delay(self, fiber_length):
        """
        Relative mode time delay correlation matrix T_mn = B_mn * L / ω.

        Parameters
        ----------
        fiber_length : float
            Length of fiber in meter (m).
        """
        corr_delay = self.corr_propconst * fiber_length / self.opt_freq
        return corr_delay

    def corr_tempcoh(self, fiber_length, tempcoh_func):
        """
        Temporal coherence mode correlation matrix γ_mn.

        Parameters
        ----------
        fiber_length : float
            Length of fiber in meter (m).
        tempcoh_func : callable
            Temporal coherence function in meter (m).
        """
        corr_tempcoh = tempcoh_func([self.corr_delay(fiber_length)])
        return corr_tempcoh

    def corr_overlap(self, overlap):
        """
        Overlap vector correlation matrix V_mn = v_m^* * v_n.

        Parameters
        ----------
        overlap : np.ndarray(1, float)
            Mode overlap vector in the same order as the internally stored
            modes.
        """
        return np.multiply.outer(np.conjugate(overlap), overlap)

    def corr_field(self, var1, var2, dvar1=0, dvar2=0, coord=None,
                   mode="linear"):
        """
        Mode output field correlation matrix evaluated at given positions
        Γ_mn (x1, x2).

        Parameters
        ----------
        var1, var2:
            Positional variables in dimensions (1, 2), which are
            typically (x, y) or (r, φ).
        dvar1, dvar2 : float
            Positional displacement w.r.t. (var1, var2) for which
            the correlation is calculated.
        coord : COORD
            Coordinate system the positional variables are given in.
        mode : str
            Displaced field calculation mode.
            "linear": First order spline interpolation.
            "cubic": Third order spline interpolation.
            "quintic": Fifth order spline interpolation.
            "exact": Re-calculation of displaced field.

        Returns
        -------
        corr_field : np.ndarray(float)
            Correlation matrix of complex mode output.
            Shape: (number of modes, shape of var_coord).
        """
        if coord is None:
            coord = self.internal_coord
        # Get mode outputs
        output1 = self.vec_field(var1, var2, coord=coord)
        output2 = output1
        if dvar1 != 0 or dvar2 != 0:
            if mode == "exact":
                output2 = self.vec_field(
                    var1 + dvar1, var2 + dvar2, coord=coord
                )
            else:
                interp = interpolate.interp2d(
                    var1, var2, output1, kind=mode  # TODO: add extrapolation
                )
                output2 = interp(var1 + dvar1, var2 + dvar2)
        corr_field = np.conjugate(output1)[:, np.newaxis] * output2
        return corr_field

    def spatial_correlation(
        self,
        var1=None, var2=None, dvar1=0, dvar2=0, coord=None,
        overlap=None, fiber_length=None, tempcoh_func=None, mode="linear"
    ):
        """
        Spatial correlation function as correlation matrix between (2D)
        coordinate vector (var1, var2).

        Parameters
        ----------
        var1, var2 : float or np.ndarray(float)
            Positional variables in dimensions (1, 2), which are
            typically (x, y) or (r, φ).
        dvar1, dvar2 : float
            Positional displacement w.r.t. (var1, var2) for which
            the correlation is calculated.
        coord : COORD
            Coordinate system the positional variables are given in.
        overlap : np.ndarray(1, float)
            Mode overlap vector in the same order as the internally stored
            modes.
        fiber_length : float
            Length of fiber in meter (m).
        tempcoh_func : callable
            Temporal coherence function in meter (m).
        mode : str
            Displaced field calculation mode.
            "nearest": Nearest neighbour interpolation.
            "linear": Linear interpolation.
            "exact": Re-calculation of displaced field.

        Returns
        -------
        spatial_correlation : np.ndarray(float)
            Spatial correlation function between given
            positional variables.
        """
        if coord is None:
            coord = self.internal_coord
        if var1 is None:
            if var2 is None:
                coord = self.internal_coord
                num = round(np.sqrt(self.mode_count) * 2)
                size = self.core_radius * 1.2 * num
                if coord == COORD.CARTESIAN:
                    var1 = np.linspace(-size, size, num=num)
                    var2 = np.linspace(-size, size, num=num)
                elif coord == COORD.POLAR:
                    var1 = np.linspace(0, size, num=num)
                    var2 = np.linspace(0, 2 * np.pi, num=num, endpoint=False)
            else:
                raise ValueError(
                    "missing var parameter for spatial correlation matrix"
                )
        if overlap is None or fiber_length is None or tempcoh_func is None:
            raise ValueError("invalid params for spatial correlation matrix")
        corr_tempcoh = self.corr_tempcoh(fiber_length, tempcoh_func)
        corr_overlap = self.corr_overlap(overlap)
        corr_field = self.corr_field(
            var1, var2, dvar1=dvar1, dvar2=dvar2, coord=coord, mode=mode
        )
        corr_field = np.moveaxis(corr_field, [0, 1], [-2, -1])
        spatial_correlation = np.sum(
            corr_tempcoh * corr_overlap * corr_field,
            axis=(-2, -1)
        )
        return spatial_correlation


# ++++++++++++++++++++++++++++++


@InheritMap(map_key=("libics-dev", "RoundStepIndexFiber"))
class RoundStepIndexFiber(Fiber):

    """
    Analytical propagation eigensystem for a round core step-index fiber.

    Parameters
    ----------
    opt_freq : float
        Optical frequency in Hertz (Hz).
    core_radius : float
        Fiber core radius in meter (m).
    clad_radius : float
        Fiber cladding radius in meter (m).
        [Not used].
    core_refr : float
        Fiber core material refraction index.
    clad_refr : float
        Fiber cladding material refraction index.
    """

    def __init__(
        self,
        opt_freq=None,
        core_radius=None, clad_radius=None,
        core_refr=None, clad_refr=None,
        pkg_name="libics-dev", cls_name="RoundStepIndexFiber", **kwargs
    ):
        super().__init__(
            coord=COORD.POLAR, opt_freq=opt_freq,
            core_radius=core_radius, clad_radius=clad_radius,
            pkg_name=pkg_name, cls_name=cls_name, **kwargs
        )
        self.core_refr = core_refr
        self.clad_refr = clad_refr
        self.core_factors = []
        self.clad_factors = []

    @property
    def internal_coord(self):
        return COORD.POLAR

    def char_eq(self, var_ev_radial, ev_azimuthal):
        """
        Calculates the characteristic equation which vanishes for
        eigenvalue solutions.

        Parameters
        ----------
        var_ev_radial
            Test value for radial eigenvalue in fiber core.
        ev_azimuthal
            Azimuthal eigenvalue.
        """
        na = self.numerical_aperture
        k0 = self.vacuum_wavenumber
        a = self.core_radius
        k = var_ev_radial
        l = ev_azimuthal    # noqa
        # Test value for radial eigenvalue in cladding
        g = np.sqrt((na * k0)**2 - k**2)
        core_term = special.jv(l + 1, k * a)
        if core_term != 0:  # finite scipy precision issues
            core_term *= k / special.jv(l, k * a)
        clad_term = special.kn(l + 1, g * a)
        if clad_term == np.inf:  # finite scipy precision issues
            clad_term = 2 * abs(core_term)
        else:
            clad_term *= g / special.kn(l, g * a)
        return core_term - clad_term

    def squared_mode_radial_core(self,
                                 radius, ev_azimuthal, ev_radial, factor):
        """
        Modulus squared of fiber core radial mode function.

        Parameters
        ----------
        radius : float
            Radial coordinate r.
        ev_azimuthal : float
            Azimuthal eigenvalue l.
        ev_radial : float
            Radial eigenvalue k.
        factor : float
            Core factor Cco.
        """
        return abs(factor * special.jv(ev_azimuthal, ev_radial * radius))**2

    def squared_mode_radial_clad(self,
                                 radius, ev_azimuthal, ev_radial, factor):
        """
        Modulus squared of fiber cladding radial mode function.

        Parameters
        ----------
        radius : float
            Radial coordinate r.
        ev_azimuthal : float
            Azimuthal eigenvalue l.
        ev_radial : float
            Radial eigenvalue k.
        factor : float
            Cladding factor Ccl.
        """
        na = self.numerical_aperture
        k0 = self.vacuum_wavenumber
        g = np.sqrt((na * k0)**2 - ev_radial**2)
        l = ev_azimuthal    # noqa
        Ccl = factor
        r = radius
        return abs(Ccl * special.kn(l, g * r))**2

    def __call__(self, var1, var2, mode_number, coord=COORD.POLAR):
        """
        Evaluates the mode profile. Does not apply any propagation phases.

        Parameters
        ----------
        var1, var2 : float or np.ndarray(float)
            Positional coordinate.
        mode_number : (int, int), int
            (l, m): (azimuthal, radial) mode number.
            (i): internal index.
        coord : COORD
            Specifies the coordinate system var_coord is given in.
            CARTESIAN: (x, y).
            POLAR: (r, φ).
        """
        r, phi, k, l, index = 5 * [None]   # noqa
        # Coordinate system
        if coord == COORD.POLAR:
            r = var1
            phi = var2
        elif coord == COORD.CARTESIAN:
            r = np.sqrt(var1**2 + var2**2)
            phi = np.arctan2(var2, var1)
        # Constants
        na = self.numerical_aperture
        k0 = self.vacuum_wavenumber
        a = self.core_radius
        nco = self.core_refr
        # Mode eigenvalues
        if isinstance(mode_number, tuple):
            index = self.mode[mode_number]
        else:
            index = mode_number
            mode_number = self.mode_numbers[index]
        b = self.mode_propconsts[index]
        k = np.sqrt((nco * k0)**2 - b**2)
        l = mode_number[0]    # noqa
        # Mode parameters
        g = np.sqrt((na * k0)**2 - k**2)
        Cco = self.core_factors[index]
        Ccl = self.clad_factors[index]
        # Evaluation
        res_azimuthal = np.exp(1j * l * phi)
        res_radial = np.piecewise(
            r, [r < a, r >= a],
            [lambda var: Cco * special.jv(l, k * var),
             lambda var: Ccl * special.kn(l, g * var)]
        )
        return res_azimuthal * res_radial

    def calc_char_eq_singularities(self, mn_l, _add_sing_count=5):
        """
        Calculates and returns the characteristic equation singularities
        up to the maximal radial eigenvalue fulfilling the guiding condition.

        Parameters
        ----------
        mn_l : int
            Azimuthal eigenvalue (or mode number).
        _add_sing_count : int
            Additional singularities to limiting estimate.
            Used recursively to include all characteristic
            equation solutions.
        """
        max_k = self.numerical_aperture * self.vacuum_wavenumber
        a = self.core_radius
        sing = (special.jn_zeros(
            mn_l,
            max(_add_sing_count,
                int(max_k * a / np.pi - mn_l / 2 + _add_sing_count))
        ) / a)
        if sing[-1] < max_k:
            sing = self.calc_char_eq_singularities(
                mn_l, _add_sing_count=(2 * _add_sing_count)
            )
        else:
            sing = np.insert(sing, 0, 0.0)
        try:
            pop_index = len(sing)
            for i in reversed(range(pop_index)):
                if sing[i - 1] >= max_k:
                    pop_index -= 1
                else:
                    break
            sing = sing[:pop_index]
        except IndexError:
            sing = []
        return sing

    def calc_modes(self):
        """
        Calculates the propagation eigensystem.
        """
        na = self.numerical_aperture
        k0 = self.vacuum_wavenumber
        a = self.core_radius
        nco = self.core_refr
        k_ls = []    # Radial eigenvalues
        mn_ls = []   # Mode numbers
        Cco_ls = []  # Core factors
        Ccl_ls = []  # Cladding factors
        # Iterate azimuthal mode number l
        l = 0   # noqa
        while True:
            # Find k intervals with exactly one solution
            k_bins = self.calc_char_eq_singularities(l)
            if len(k_bins) == 0:
                break
            # Iterate radial mode number m(l)
            k, m = None, None
            for i, k_bin_left in enumerate(k_bins[:-1]):
                k_bin_right = min(k_bins[i + 1], na * k0)
                # Find k eigenvalue by finding roots of characteristic equation
                try:
                    # brentq algorithm guarantees convergence
                    k = optimize.brentq(
                        self.char_eq,
                        k_bin_left + 1e-5, k_bin_right - 1e-5,
                        args=(l, )
                    )
                except ValueError:
                    break
                m = i + 1
                # Demand continuity at core-cladding interface
                Cco = 1.0
                Ccl = (special.jv(l, k * a)
                       / special.kn(l, np.sqrt((na * k0)**2 - k**2) * a))
                # Append to result lists
                k_ls.append(k)
                mn_ls.append((l, m))
                Cco_ls.append(Cco)
                Ccl_ls.append(Ccl)
                if l != 0:  # noqa  # Using l/-l symmetry
                    k_ls.append(k)
                    mn_ls.append((-l, m))
                    Cco_ls.append(Cco)
                    Ccl_ls.append(Ccl if l % 2 == 0 else -Ccl)
            if m is None:
                break
            l += 1  # noqa
        # Iterate results
        for i, (l, m) in enumerate(mn_ls):
            # Calculate normalization (2π from polar integration)
            core_integral = 2 * np.pi * integrate.fixed_quad(
                self.squared_mode_radial_core, 0, a,
                args=(l, k_ls[i], 1),
                n=round(na * k0 * a - abs(l) / 2)
            )[0]
            clad_integral = 2 * np.pi * integrate.quad(
                self.squared_mode_radial_clad, a, np.inf,
                args=(l, k_ls[i], Ccl_ls[i]),
                limit=1000
            )[0]
            normalization = np.sqrt(core_integral + clad_integral)
            Cco_ls[i] /= normalization
            Ccl_ls[i] /= normalization
        # Assign results
        self.mode_propconsts = np.sqrt((nco * k0)**2 - np.array(k_ls)**2)
        self.mode_numbers = np.array(mn_ls)
        self.core_factors = np.array(Cco_ls)
        self.clad_factors = np.array(Ccl_ls)
        self._set_mode_number_map()

    def calc_overlap(
        self,
        input_field, *args,
        algorithm="dblquad", **kwargs
    ):
        """
        Parameters
        ----------
        input_field : callable
            Function representing the input light field.
            Call signature : f (var_1, var_2, coord)
                var_1, var_2 : float
                    2D spatial variables.
                coord : COORD
                    Coordinate system type (must accept polar).
        *args
            Parameters passed to input_field.
        algorithm : "dblquad", "hybrid", "simpson"
            Overlap integration algorithm.
        **kwargs
            Keyword arguments passed to input_field.

        Returns
        -------
        overlap : numpy.ndarray(1)
            Overlap vector corresponding to the fiber modes.
        """
        # Perform overlap integral
        print("Calculating overlap integral")
        overlap = np.full_like(self.mode_propconsts, np.nan, dtype=complex)
        t0 = time.time()

        # Simpsons integration
        if algorithm == "simpson":
            for i, (l, m) in enumerate(self.mode_numbers):
                print(
                    "\r{: >6d}/{:d} ({:d}s)      "
                    .format(i, len(overlap), int(time.time() - t0)),
                    end=""
                )
                var_azimuthal = np.linspace(
                    0, 2 * np.pi, num=max(16, 8 * abs(l)), endpoint=False
                )
                var_radial = np.linspace(
                    0, 1.2 * self.core_radius,
                    num=int(max(24, 6 * m)),
                    endpoint=False
                )
                r, phi = np.meshgrid(var_radial, var_azimuthal)
                overlap[i] = 2 * np.pi * integrate.simps(
                    integrate.simps(np.conjugate(self(r, phi, i))
                                    * input_field(r, phi, coord=COORD.POLAR),
                                    var_radial),
                    var_azimuthal
                )

        # Hybrid integration
        if algorithm == "hybrid":
            for i, (l, m) in enumerate(self.mode_numbers):
                print(
                    "\r{: >4d}/{:d} ({:d}s)"
                    .format(i, len(overlap), int(time.time() - t0)),
                    end=""
                )
                var_azimuthal = np.linspace(
                    0, 2 * np.pi, num=max(16, 8 * abs(l)), endpoint=False
                )
                # Integrate core
                overlap[i] = integrate.simps(
                    [integrate.fixed_quad(
                        lambda r: np.real(
                            np.conjugate(self(r, phi, i))
                            * input_field(r, phi, coord=COORD.POLAR)
                        ), 0, self.core_radius, n=max(6, 6 * m)
                    )[0] for phi in var_azimuthal],
                    x=var_azimuthal
                )
                overlap[i] += 1j * integrate.simps(
                    [integrate.fixed_quad(
                        lambda r: np.imag(
                            np.conjugate(self(r, phi, i))
                            * input_field(r, phi, coord=COORD.POLAR)
                        ), 0, self.core_radius, n=max(6, 6 * m)
                    )[0] for phi in var_azimuthal],
                    x=var_azimuthal
                )
                # Integrate cladding
                overlap[i] += integrate.simps(
                    [integrate.fixed_quad(
                        lambda r: np.real(
                            np.conjugate(self(r, phi, i))
                            * input_field(r, phi, coord=COORD.POLAR)
                        ),
                        self.core_radius, 1.2 * self.core_radius,
                        n=max(6, 6 * m)
                    )[0] for phi in var_azimuthal],
                    x=var_azimuthal
                )
                overlap[i] += 1j * integrate.simps(
                    [integrate.fixed_quad(
                        lambda r: np.imag(
                            np.conjugate(self(r, phi, i))
                            * input_field(r, phi, coord=COORD.POLAR)
                        ),
                        self.core_radius, 1.2 * self.core_radius,
                        n=max(6, 6 * m)
                    )[0] for phi in var_azimuthal],
                    x=var_azimuthal
                )
                overlap[i] *= 2 * np.pi

        # Library quad integration
        if algorithm == "dblquad":
            for i, (l, m) in enumerate(self.mode_numbers):
                print(
                    "\r{: >4d}/{:d} ({:d}s)"
                    .format(i, len(overlap), int(time.time() - t0)),
                    end=""
                )
                # Integrate core
                overlap[i] = integrate.dblquad(
                    lambda r, phi: np.real(
                        np.conjugate(self(r, phi, i))
                        * input_field(r, phi, coord=COORD.POLAR)
                    ),
                    0, 2 * np.pi,
                    0, self.core_radius,
                )[0]
                overlap[i] += 1j * integrate.dblquad(
                    lambda r, phi: np.imag(
                        np.conjugate(self(r, phi, i))
                        * input_field(r, phi, coord=COORD.POLAR)
                    ),
                    0, 2 * np.pi,
                    0, self.core_radius
                )[0]
                # Integrate cladding
                overlap[i] += integrate.dblquad(
                    lambda r, phi: np.real(
                        np.conjugate(self(r, phi, i))
                        * input_field(r, phi, coord=COORD.POLAR)
                    ),
                    0, 2 * np.pi,
                    self.core_radius, 1.5 * self.core_radius
                )[0]
                overlap[i] += 1j * integrate.dblquad(
                    lambda r, phi: np.imag(
                        np.conjugate(self(r, phi, i))
                        * input_field(r, phi, coord=COORD.POLAR)
                    ),
                    0, 2 * np.pi,
                    self.core_radius, 1.5 * self.core_radius
                )[0]
                overlap[i] *= 2 * np.pi
        print(
            "\r{: >6d}/{:d} ({:d}s)"
            .format(len(overlap), len(overlap), int(time.time() - t0))
        )
        return overlap
