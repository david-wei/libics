import abc
import os
import time

import numpy as np
from scipy import integrate, constants, optimize, special

from libics import env
from libics.file import hdf
from libics.util import InheritMap


###############################################################################


class COORD:

    CARTESIAN = "cartesian"
    POLAR = "polar"


@InheritMap(map_key=("libics-dev", "Fiber"))
class Fiber(abc.ABC, hdf.HDFBase):

    """
    Fiber propagation eigensystem with analytical modes.

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

    def _set_mode_number_map(self):
        self.__mode_number_map = {}
        for i, mn in enumerate(self.mode_numbers):
            self.__mode_number_map[tuple(mn)] = i

    def _hdf_init_write(self):
        self.__mode_number_map = {}

    @property
    def mode(self):
        if not self.__mode_number_map:
            self._set_mode_number_map()
        return self.__mode_number_map

    @property
    def mode_count(self):
        return len(self.mode_numbers)

    @property
    def mode_numbers(self):
        return self._mode_numbers

    @mode_numbers.setter
    def mode_numbers(self, val):
        self._mode_numbers = val
        self._set_mode_number_map()

    @abc.abstractmethod
    def __call__(*coords):
        """
        Evaluates the complex mode profile.

        Implementations should take coordinate and optional parameters.
        """
        pass


# ++++++++++++++++++++++++++++++


@InheritMap(map_key=("libics-dev", "RoundStepIndexFiber"))
class RoundStepIndexFiber(Fiber):

    """
    Propagation eigensystem for a round core step-index fiber.

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
    def numerical_aperture(self):
        return np.sqrt(self.core_refr**2 - self.clad_refr**2)

    @property
    def vacuum_wavenumber(self):
        return 2 * np.pi * self.opt_freq / constants.speed_of_light

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

    def __call__(self, radius, azimuth, mode_number, coord=COORD.POLAR):
        """
        Evaluates the mode profile. Does not apply any propagation phases.

        Parameters
        ----------
        radius : float
            Radial coordinate.
        azimuth : float
            Polar coordinate.
        mode_number : (int, int), int
            (l, m): (azimuthal, radial) mode number.
            (i): internal index.
        coord : COORD
            Specifies given coordinate system.
            If COORD.CARTESIAN, the parameters (radius, azimuth)
            are interpreted as (x, y).
        """
        r, phi, k, l, index = 5 * [None]   # noqa
        # Coordinate system
        if coord == COORD.POLAR:
            r = radius
            phi = azimuth
        elif coord == COORD.CARTESIAN:
            r = np.sqrt(radius**2 + azimuth**2)
            phi = np.arctan2(azimuth, radius)
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
        coord=COORD.CARTESIAN, **kwargs
    ):
        """
        Parameters
        ----------
        input_field : callable
            Function representing the input light field.
        *args
            Parameters passed to input_field.
        coord : COORD
            Coordinate system of input_field.
        **kwargs
            Keyword arguments passed to input_field.

        Returns
        -------
        overlap : numpy.ndarray(1)
            Overlap vector corresponding to the fiber modes.
        """
        _input_field = input_field
        # Convert input_field to take polar coordinates
        if coord == COORD.CARTESIAN:
            input_field = np.frompyfunc(
                lambda r, phi: _input_field(
                    r * np.cos(phi), r * np.sin(phi), *args, **kwargs
                ), 2, 1
            )
        # Perform overlap integral
        print("Calculating overlap integral")
        overlap = np.full_like(self.mode_propconsts, np.nan, dtype=complex)
        t0 = time.time()
        for i, item in enumerate(overlap):
            print("\r{: >4d}/{:d} ({:d}s)"
                  .format(i + 1, len(overlap), int(time.time() - t0)), end="")
            # Integrate core
            overlap[i] = integrate.dblquad(
                lambda r, phi:
                np.real(np.conjugate(self(r, phi, i)) * input_field(r, phi)),
                0, 2 * np.pi,
                0, self.core_radius,
            )[0]
            overlap[i] += 1j * integrate.dblquad(
                lambda r, phi:
                np.imag(np.conjugate(self(r, phi, i)) * input_field(r, phi)),
                0, 2 * np.pi,
                0, self.core_radius
            )[0]
            # Integrate cladding
            overlap[i] += integrate.dblquad(
                lambda r, phi:
                np.real(np.conjugate(self(r, phi, i)) * input_field(r, phi)),
                0, 2 * np.pi,
                self.core_radius, 1.5 * self.core_radius
            )[0]
            overlap[i] += 1j * integrate.dblquad(
                lambda r, phi:
                np.imag(np.conjugate(self(r, phi, i)) * input_field(r, phi)),
                0, 2 * np.pi,
                self.core_radius, 1.5 * self.core_radius
            )[0]
        print()
        return overlap

    def output(self, radius, azimuth, overlap, length, coord=COORD.POLAR):
        """
        Parameters
        ----------
        radius : float
            Radial coordinate.
        azimuth : float
            Polar coordinate.
        overlap : numpy.ndarray(1)
            Overlap vector corresponding to the fiber modes.
        length : float
            Propagation length in meter (m).
        coord : COORD
            Specifies given coordinate system.
            If COORD.CARTESIAN, the parameters (radius, azimuth)
            are interpreted as (x, y).
        """
        r, phi = 2 * [None]   # noqa
        # Coordinate system
        if coord == COORD.POLAR:
            r = radius
            phi = azimuth
        elif coord == COORD.CARTESIAN:
            r = np.sqrt(radius**2 + azimuth**2)
            phi = np.arctan2(azimuth, radius)
        # Output of each mode
        mode_profile = np.array(
            [self(r, phi, index) for index in np.arange(self.mode_count)]
        )
        # Sum up modes
        output = np.sum(
            overlap
            * np.exp(-1j * self.mode_propconsts * length)
            * np.moveaxis(mode_profile, 0, -1),
            axis=-1
        )
        return output


###############################################################################


@InheritMap(map_key=("libics-dev", "GaussianBeam"))
class GaussianBeam(hdf.HDFBase):

    """
    Parameters
    ----------
    opt_freq : float
        Optical frequency in Hertz (Hz).
    waist : float
        Gaussian beam waist in meter (m).
    rotation_x, rotation_y, rotation_z : float
        Rotation of beam incidence in radians (rad)
        with x, y, z axis as rotation axis.
    rotation : (float, float, float)
        (x, y, z) rotation angles. Overwrites element-wise parameters.
    offset_x, offset_y, offset_z : float
        Offset in x, y, z direction in meter (m).
    offset : (float, float, float)
        (x, y, z) offsets. Overwrites element-wise parameters.
    """

    def __init__(
        self,
        opt_freq=(constants.speed_of_light / 780e-9), waist=100e-6,
        rotation_x=0.0, rotation_y=0.0, rotation_z=0.0, rotation=None,
        offset_x=0.0, offset_y=0.0, offset_z=0.0, offset=None
    ):
        super().__init__(pkg_name="libics-dev", cls_name="GaussianBeam")
        self.opt_freq = opt_freq
        self.waist = waist
        if rotation is None:
            rotation = np.array([rotation_x, rotation_y, rotation_z])
        self.rotation = np.array(rotation)
        if offset is None:
            offset = np.array([offset_x, offset_y, offset_z])
        self.offset = np.array(offset)

    @property
    def vacuum_wavenumber(self):
        return 2 * np.pi * self.opt_freq / constants.speed_of_light

    @property
    def vacuum_wavelength(self):
        return constants.speed_of_light / self.opt_freq

    @property
    def rayleigh_range(self):
        return np.pi * self.waist / self.vacuum_wavelength

    def cv_coordinates(self, x, y, z):
        """
        Performs inverse coordinate system rotation and translation
        as specified by the offset and rotation attributes.
        """
        vec = np.array([x, y, z])
        ra = self.rotation
        vec_rotated = np.matmul(
            np.matmul(
                np.array([
                    [1, 0, 0],
                    [0, np.cos(ra[0]), -np.sin(ra[0])],
                    [0, np.sin(ra[0]), np.cos(ra[0])]
                ]),
                np.array([
                    [np.cos(ra[0]), 0, np.sin(ra[0])],
                    [0, 1, 0],
                    [-np.sin(ra[0]), 0, np.cos(ra[0])]
                ])
            ),
            np.matmul(
                np.array([
                    [np.cos(ra[0]), -np.sin(ra[0]), 0],
                    [np.sin(ra[0]), np.cos(ra[0]), 0],
                    [0, 0, 1]
                ]),
                vec
            )
        )
        vec_translated = vec_rotated - self.offset
        return vec_translated

    def __call__(self, x, y, z=0.0):
        """
        Calculates the Gaussian beam field with stored transformations.
        """
        xx, yy, zz = self.cv_coordinates(x, y, z)
        r = np.sqrt(xx**2 + yy**2)
        k = self.vacuum_wavenumber
        w0 = self.waist
        zR = self.rayleigh_range
        wn = np.sqrt(1 + (zz / zR)**2)

        divergence = 1
        if zz != 0:
            R = zz + zR**2 / zz
            divergence = np.exp(-1j * k * r**2 / 2 / R)

        res = (
            1 / wn * np.exp(-(r / w0 / wn)**2)
            * divergence
            * np.exp(-1j * (k * zz + np.arctan(zz / zR)))
        )
        return res.astype(complex)


###############################################################################


@InheritMap(map_key=("libics-dev", "OverlapFiberBeam"))
class OverlapFiberBeam(hdf.HDFBase):

    def __init__(self, beam=None, fiber=None, overlap=None):
        self.beam = beam
        self.fiber = fiber
        self.overlap = overlap

    def calc_modes(self):
        """
        Calculates the fiber modes.
        """
        self.fiber.calc_modes()

    def calc_overlap(self):
        """
        Calculates the overlap based on the stored beam and fiber.
        """
        self.overlap = self.fiber.calc_overlap(
            self.beam, coord=COORD.CARTESIAN
        )

    def field(self, radius=None, points=250, length=1):
        """
        Gets a sampled fiber output field.

        Parameters
        ----------
        radius : float
            Sampling radius [-r, r] in meter (m).
        points : int
            Number of sampling points in one dimension.
        length : float
            Propagation length in meter (m).
        """
        if radius is None:
            radius = 1.1 * self.fiber.core_radius
        r = np.linspace(-radius, radius, num=points)
        x, y = np.meshgrid(r, r)
        field = self.fiber.output(
            x, y, self.overlap, length, coord=COORD.CARTESIAN
        )
        return field

    def intensity(self, radius=None, points=250, length=1):
        """
        Gets a sampled fiber output intensity.

        Parameters
        ----------
        radius : float
            Sampling radius [-r, r] in meter (m).
        points : int
            Number of sampling points in one dimension.
        length : float
            Propagation length in meter (m).
        """
        field = self.field(radius=radius, points=points, length=length)
        return np.abs(field)**2

    def correlation(self, displacement, radius=None, points=250, length=1):
        """
        Gets a sampled fiber output correlation.

        Parameters
        ----------
        displacement : (float, float)
            Correlation displacement (x, y) in meter (m).
        radius : float
            Sampling radius [-r, r] in meter (m).
        points : int
            Number of sampling points in one dimension.
        length : float
            Propagation length in meter (m).
        """
        if radius is None:
            radius = 1.1 * self.fiber.core_radius
        r = np.linspace(-radius, radius, num=points)
        x, y = np.meshgrid(r, r)
        field_orig = self.fiber.output(
            x, y, self.overlap, length, coord=COORD.CARTESIAN
        )
        field_disp = self.fiber.output(
            x + displacement[0], y + displacement[1],
            self.overlap, length, coord=COORD.CARTESIAN
        )
        correlation = np.conjugate(field_orig) * field_disp
        return correlation

    def coherence(self, displacement, radius=None, points=250, length=1):
        """
        Gets a sampled fiber output coherence.

        Parameters
        ----------
        displacement : (float, float)
            Correlation displacement (x, y) in meter (m).
        radius : float
            Sampling radius [-r, r] in meter (m).
        points : int
            Number of sampling points in one dimension.
        length : float
            Propagation length in meter (m).
        """
        if radius is None:
            radius = 1.1 * self.fiber.core_radius
        r = np.linspace(-radius, radius, num=points)
        x, y = np.meshgrid(r, r)
        field_orig = self.fiber.output(
            x, y, self.overlap, length, coord=COORD.CARTESIAN
        )
        field_disp = self.fiber.output(
            x + displacement[0], y + displacement[1],
            self.overlap, length, coord=COORD.CARTESIAN
        )
        correlation = np.conjugate(field_orig) * field_disp
        mean_intensity = np.abs(field_orig) * np.abs(field_disp)
        coherence = correlation / mean_intensity
        return coherence


###############################################################################


def plot_overlap(overlap, length=1, data="field", **kwargs):
    """
    data : "field", "intensity", "correlation", "coherence"
    length : float
    **kwargs
        displacement : float
    """
    dtype = data
    import matplotlib.pyplot as plt
    if dtype == "field":
        data = np.real(overlap.field(length=length))
    elif dtype == "intensity":
        data = overlap.intensity(length=length)
    elif dtype == "correlation":
        data = np.abs(overlap.correlation(kwargs["displacement"],
                                          length=length))
    elif dtype == "coherence":
        data = np.abs(overlap.coherence(kwargs["displacement"], length=length))
    vmax, vmin, cmap = 1, 0, "viridis"
    if dtype != "coherence":
        vmax = max(data.max(), -data.min())
        vmin = -vmax
        cmap = "RdBu"
        if dtype == "intensity":
            vmin = 0
            cmap = "Oranges"
    plt.pcolormesh(data, cmap=cmap, vmin=vmin, vmax=vmax)
    title = dtype + " $"
    title += "w_0 = {:.0f} μm, ".format(overlap.beam.waist * 1e6)
    title += "\\theta = {:.0f}°, ".format(
        np.linalg.norm(overlap.beam.rotation) * 180 / np.pi
    )
    title += "\\Delta x_0 = {:.0f} μm, ".format(
        np.linalg.norm(overlap.beam.offset) * 1e6
    )
    title += "$NA $= {:.2f}, ".format(overlap.fiber.numerical_aperture)
    title += "a = {:.0f} µm, ".format(overlap.fiber.core_radius * 1e6)
    title += "L = {:.1f} m".format(length)
    if dtype == "coherence" or dtype == "correlation":
        title += ", \\Delta x = ({:.0f}, {:.0f}) µm$".format(
            kwargs["displacement"][0] * 1e6, kwargs["displacement"][1] * 1e6
        )
    else:
        title += "$"
    plt.title(title)
    plt.gca().set_aspect("equal")
    plt.colorbar()
    plt.show()


def rsif_main():
    core_refr = 1.50
    clad_refr = 1.4838
    core_radius = 5e-6
    wavelength = 780e-9
    plot_mode_number = (0, 2)
    plot_linear_pixels = 250

    import cProfile
    profiler = cProfile.Profile()

    mmf = RoundStepIndexFiber(
        opt_freq=(constants.speed_of_light / wavelength),
        core_radius=core_radius,
        core_refr=core_refr, clad_refr=clad_refr
    )
    profiler.enable()
    mmf.calc_modes()
    profiler.disable()
    profiler.print_stats(sort="time")

    import matplotlib.pyplot as plt
    # Check continuity condition
    """
    r = np.linspace(0.95, 1.05, plot_linear_pixels) * core_radius
    for index in range(mmf.mode_count):
        plt.plot(
            r / core_radius,
            np.real(mmf(r, 0, index, coord=COORD.POLAR)),
            label=str(mmf.mode_numbers[index])
        )
    for index in range(mmf.mode_count):
        plt.plot(
            r / core_radius,
            np.imag(mmf(r, 0, index, coord=COORD.POLAR)),
            label=str(mmf.mode_numbers[index])
        )
    plt.xlabel("Radius $r / a$")
    plt.legend()
    plt.show()
    """
    # Check mode profiles
    r = np.linspace(-1.1 * core_radius, 1.1 * core_radius, plot_linear_pixels)
    xx, yy = np.meshgrid(r, r)
    zz = mmf(xx, yy, plot_mode_number, coord=COORD.CARTESIAN)
    plt_zz = np.real(zz)
    vmax = max(plt_zz.max(), -plt_zz.min())
    vmin = -vmax
    plt.pcolormesh(plt_zz, cmap="RdBu", vmin=vmin, vmax=vmax)
    title = "Re[Mode] (total: {:d}): ".format(mmf.mode_count)
    title += "$\\left|l, m\\right> = "
    title += "\\left|{:d}, {:d}\\right>$, ".format(*plot_mode_number)
    title += "$n_{co} " + "k_0 / \\beta = {:.3f}$".format(
        mmf.core_refr * mmf.vacuum_wavenumber
        / mmf.mode_propconsts[mmf.mode[plot_mode_number]]
    )
    plt.title(title)
    plt.gca().set_aspect("equal")
    plt.colorbar()
    plt.show()

    return mmf


def gb_main():
    wavelength = 780e-9
    propagation_angle = 1e-2
    waist = 100e-6
    offset_x = 0.0
    offset_y = 0.0
    plot_linear_pixels = 250
    plot_position_z = 0.0

    gb = GaussianBeam(
        opt_freq=(constants.speed_of_light / wavelength),
        propagation_angle=propagation_angle, waist=waist,
        offset_x=offset_x, offset_y=offset_y
    )

    import matplotlib.pyplot as plt
    r = np.linspace(-1.5 * waist, 1.5 * waist, plot_linear_pixels)
    xx, yy = np.meshgrid(r, r)
    zz = gb(xx, yy, plot_position_z)
    plt_zz = np.real(zz)
    plt.pcolormesh(plt_zz, cmap="RdBu")
    title = "Re[Gaussian beam]: $w_0 = {:.0f} μm, ".format(gb.waist * 1e6)
    title += ("\\theta = {:.0f}°, (x_0, y_0) = ({:.0f}, {:.0f}) μm$"
              .format(gb.propagation_angle * 180 / np.pi,
                      gb.offset_x * 1e6, gb.offset_y * 1e6))
    plt.title(title)
    plt.gca().set_aspect("equal")
    plt.colorbar()
    plt.show()


def prop_main():
    core_refr = 1.50
    clad_refr = 1.4838
    core_radius = 52.5e-6
    prop_length = 20.0
    wavelength = 780e-9
    waist = 20e-6
    rotation = [1e-1, 0.0, 0.0]
    offset = [0.0, 0.0, 0.0]
    file_path = os.path.join(env.DIRS["rzgdatashare"], "RZG_libics",
                             "mmf_propagation", "overlap_1.hdf5")

    gb = GaussianBeam(
        opt_freq=(constants.speed_of_light / wavelength), waist=waist,
        offset=offset, rotation=rotation
    )
    mmf = RoundStepIndexFiber(
        opt_freq=(constants.speed_of_light / wavelength),
        core_radius=core_radius,
        core_refr=core_refr, clad_refr=clad_refr
    )
    overlap = OverlapFiberBeam(beam=gb, fiber=mmf)
    overlap.calc_modes()
    overlap.calc_overlap()
    hdf.write_hdf(overlap, file_path=file_path)
    plot_overlap(overlap, length=prop_length, data="field")


def overlap_load():
    length = 1
    dx = (10e-6, 0)
    file_path = os.path.join(env.DIRS["rzgdatashare"], "RZG_libics",
                             "mmf_propagation", "overlap_1.hdf5")
    overlap = hdf.read_hdf(OverlapFiberBeam, file_path=file_path)
    plot_overlap(overlap, length=length, data="field")
    plot_overlap(overlap, length=length, data="intensity")
    plot_overlap(overlap, length=length, data="correlation", displacement=dx)
    plot_overlap(overlap, length=length, data="coherence", displacement=dx)


def monocoh_main():
    """monochromatic coherence"""
    # Fiber
    core_refr = 1.50
    clad_refr = 1.4838
    core_radius = 10e-6
    prop_length = 20.0
    # Light source
    wavelength = 780e-9
    propagation_angle = 1e-1
    waist = 3e-6
    offset_x = 0.0
    offset_y = 0.0
    offset_z = 0.0
    # Coherence
    coh_dx = 10e-6
    coh_dy = 0.0
    # Plotting
    plot_linear_pixels = 250

    gb = GaussianBeam(
        opt_freq=(constants.speed_of_light / wavelength),
        propagation_angle=propagation_angle, waist=waist,
        offset_x=offset_x, offset_y=offset_y
    )
    mmf = RoundStepIndexFiber(
        opt_freq=(constants.speed_of_light / wavelength),
        core_radius=core_radius,
        core_refr=core_refr, clad_refr=clad_refr
    )
    mmf.calc_modes()
    overlap = mmf.calc_overlap(gb.__call__, offset_z, coord=COORD.CARTESIAN)

    import matplotlib.pyplot as plt
    # Check coherence as function of (x, y) at fixed displacement (dx, dy)
    r = np.linspace(-1.1 * core_radius, 1.1 * core_radius, plot_linear_pixels)
    xx, yy = np.meshgrid(r, r)
    output_1 = mmf.output(xx, yy, overlap,
                          prop_length, coord=COORD.CARTESIAN)
    output_2 = mmf.output(xx + coh_dx, yy + coh_dy, overlap,
                          prop_length, coord=COORD.CARTESIAN)
    correlation = np.conjugate(output_1) * output_2
    magnitude_1 = np.abs(output_1)
    magnitude_2 = np.abs(output_2)
    coherence = correlation / magnitude_1 / magnitude_2
    deg_coh = np.abs(coherence)
    plt.pcolormesh(deg_coh, cmap="viridis", vmin=0, vmax=1)
    title = ("|Coherence|: $\\Delta x = ({:.0f}, {:.0f}) µm, w_0 = {:.0f} μm, "
             .format(coh_dx * 1e6, coh_dy * 1e6, gb.waist * 1e6))
    title += ("\\theta = {:.0f}°, \Delta \\rho = {:.0f} μm, $"
              .format(gb.propagation_angle * 180 / np.pi,
                      np.sqrt(gb.offset_x**2 + gb.offset_y**2) * 1e6))
    title += ("NA $= {:.2f}, a = {:.0f} µm, L = {:.1f} m$".format(
        mmf.numerical_aperture, mmf.core_radius * 1e6, prop_length
    ))
    plt.title(title)
    plt.gca().set_aspect("equal")
    plt.colorbar()
    plt.show()


###############################################################################


if __name__ == "__main__":
    # gb_main()
    # rsif_main()
    prop_main()
    overlap_load()
    # monocoh_main()
