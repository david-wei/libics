import abc

import numpy as np
from scipy import integrate, constants, optimize, special

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

    @property
    def mode(self):
        return self.__mode_number_map

    @property
    def mode_count(self):
        return len(self.mode_numbers)

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
        Evaluates the mode profile.

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
            b = self.mode_propconsts[index]
            k = np.sqrt((nco * k0)**2 - b**2)
            l = mode_number[0]    # noqa
        else:
            index = mode_number
            b = self.mode_propconsts[index]
            k = np.sqrt((nco * k0)**2 - b**2)
            l = self.mode_numbers[mode_number[0]]    # noqa
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
                    Ccl_ls.append(Ccl)
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


###############################################################################


def main():
    core_refr = 1.50
    clad_refr = 1.4838
    core_radius = 52.5e-6
    wavelength = 780e-9
    plot_mode_number = (-3, 2)
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
    r = np.linspace(-1.1 * core_radius, 1.1 * core_radius, plot_linear_pixels)
    xx, yy = np.meshgrid(r, r)
    zz = mmf(xx, yy, plot_mode_number, coord=COORD.CARTESIAN)
    plt_zz = np.real(zz)**2
    plt.pcolormesh(plt_zz)
    title = "Re[Mode]² (total: {:d}): ".format(mmf.mode_count)
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


###############################################################################


if __name__ == "__main__":
    main()
