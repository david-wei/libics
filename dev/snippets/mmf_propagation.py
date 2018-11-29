import numpy as np
from scipy import special, integrate, constants, optimize


###############################################################################


class COORD:

    CARTESIAN = "cartesian"
    POLAR = "polar"


class Fiber(object):

    """analytical modes"""

    def __init__(
        self,
        coord=COORD.CARTESIAN, opt_freq=None,
        core_radius=None, clad_radius=None,
        mode_numbers=[], mode_propconsts=[], mode_functions=[]
    ):
        self.coord = coord
        self.opt_freq = opt_freq
        self.core_radius = core_radius
        self.clad_radius = clad_radius
        self.mode_numbers = mode_numbers
        self.mode_propconsts = mode_propconsts
        self.mode_functions = mode_functions
        self.__mode_number_map = {}

    def _set_mode_number_map(self):
        self.__mode_number_map = {}
        for i, mn in enumerate(self.mode_numbers):
            self.__mode_number_map[mn] = i

    @property
    def mode(self):
        return self.__mode_number_map


# ++++++++++++++++++++++++++++++


class RoundStepIndexFiber(Fiber):

    def __init__(
        self,
        opt_freq=None,
        core_radius=None, clad_radius=None,
        core_refr=None, clad_refr=None
    ):
        super().__init__(
            coord=COORD.POLAR, opt_freq=opt_freq,
            core_radius=core_radius, clad_radius=clad_radius
        )
        self.clad_cont_scales = []
        self.core_refr = core_refr
        self.clad_refr = clad_refr
        self._na = np.sqrt(self.core_refr**2 - self.clad_refr**2)
        self._k0 = 2 * np.pi * self.opt_freq / constants.speed_of_light

    def char_eq(self, k, l):
        """characteristic equation"""
        g = np.sqrt((self._na * self._k0)**2 - k**2)
        return (
            k * special.jv(l + 1, k * self.core_radius)
            / special.jv(l, k * self.core_radius)
            - g * special.kn(l + 1, g * self.core_radius)
            / special.kn(l, g * self.core_radius)
        )

    def mode_radial_core(self, x, l, k_lm, scale):
        """modulus squared core radial mode function"""
        return abs(scale * special.jv(l, k_lm * x))**2

    def mode_radial_clad(self, x, l, k_lm, scale):
        """modulus squared cladding radial mode function"""
        return abs(scale * special.kn(
            l,
            np.sqrt((self._na * self._k0 * self.core_radius)**2 - k_lm**2) * x
        ))**2

    def calc_modes(self):
        """calculate eigensystem"""
        # 0. Utility variables
        self.mode_numbers = []
        self.mode_propconsts = []
        self.mode_functions = []
        self.clad_cont_scales = []
        na = self._na
        k0 = self._k0
        a = self.core_radius
        k_lms = []
        # Iterate azimuthal mode number l
        l = 0   # noqa
        cont_l = True
        while cont_l:
            k_bins = np.insert(
                special.jn_zeros(
                    l, int(na * k0 * a / np.pi - l / 2 + 5)
                ) / a,   # +5 to include all char. eq. sol. singularities
                0, 0.0
            )
            # Iterate radial mode number m(l)
            m = 0
            for i, k_bin_start in enumerate(k_bins):
                m += 1
                # Break if m(l) value would lead to unguided mode
                if k_bin_start > na * k0:
                    m -= 1
                    break
                k_bin_end = min(k_bins[i + 1], na * k0)
                # Calculate propagation constant
                try:
                    k_lm = optimize.brentq(
                        self.char_eq,
                        k_bin_start + 1e-5, k_bin_end - 1e-5,
                        args=(l, )
                    )
                except ValueError:
                    m -= 1
                    break
                prop_const = np.sqrt((self.core_refr * self._k0)**2 - k_lm**2)
                # Apply continuity
                clad_cont_scale = (
                    special.jv(l, k_lm * a)
                    / special.kn(l, np.sqrt((na * k0)**2 - k_lm**2) * a)
                )
                # Append result lists
                k_lms.append(k_lm)
                self.mode_numbers.append((l, m))
                self.mode_propconsts.append(prop_const)
                self.clad_cont_scales.append(clad_cont_scale)
                if l != 0:  # noqa  # Using l/-l symmetry
                    k_lms.append(k_lm)
                    self.mode_numbers.append((-l, m))
                    self.mode_propconsts.append(prop_const)
                    self.clad_cont_scales.append(clad_cont_scale)
            if m > 0:
                l += 1  # noqa
            else:
                cont_l = False
        # Iterate results
        for i, (l, m) in enumerate(self.mode_numbers):
            # Calculate normalization
            # (2π from polar integration)
            core_integral = 2 * np.pi * integrate.fixed_quad(
                self.mode_radial_core, 0, a,
                args=(l, k_lms[i], 1),
                n=round(na * k0 * a - abs(l) / 2)
            )[0]
            clad_integral = 2 * np.pi * integrate.quad(
                self.mode_radial_clad, a, np.inf,
                args=(l, k_lms[i], self.clad_cont_scales[i]),
                limit=1000
            )[0]
            normalization = np.sqrt(core_integral + clad_integral)

            # Construct mode functions
            def f(r, phi):
                return np.exp(1j * l * phi) / normalization * np.piecewise(
                    r, [r < a, r >= a], [
                        special.jv(l, k_lms[i] * r),
                        self.clad_cont_scales[i] * special.kn(
                            l, np.sqrt((na * k0)**2 - k_lms[i]**2) * r
                        )
                    ]
                )
            self.mode_functions.append(np.frompyfunc(f, 2, 1))
        self._set_mode_number_map()

    def mode_function(self, r, phi, mode):
        """complex mode profile, mode = (l, m) or index"""
        if not isinstance(mode, tuple):
            mode = self.mode_numbers[mode]
        index = self.mode[mode]
        l, m = mode
        na, k0, a = self._na, self._k0, self.core_radius
        

        return np.exp(1j * l * phi) / normalization * np.piecewise(
            r, [r < a, r >= a], [
                special.jv(l, k_lms[i] * r),
                self.clad_cont_scales[i] * special.kn(
                    l, np.sqrt((na * k0)**2 - k_lms[i]**2) * r
                )
            ]
        )



###############################################################################


def main():
    core_refr = 1.50
    clad_refr = 1.4838
    # core_radius = 52.5
    core_radius = 5.0e-6
    wavelength = 780e-9

    mmf = RoundStepIndexFiber(
        opt_freq=(constants.speed_of_light / wavelength),
        core_radius=core_radius,
        core_refr=core_refr, clad_refr=clad_refr
    )
    mmf.calc_modes()
    return mmf


###############################################################################


if __name__ == "__main__":
    main()
