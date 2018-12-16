import numpy as np
from scipy import constants

from libics.data import arraydata, seriesdata
from libics.file import hdf
from libics.util import InheritMap
from libics.trafo.data import cv_seriesdata_to_arraydata
from libics.dev.sim.mmf_modes import COORD

###############################################################################


@InheritMap(map_key=("libics-dev", "GaussianBeam"))
class GaussianBeam(hdf.HDFBase):

    """
    Parameters
    ----------
    spectrum : float or data.arraydata.ArrayData or data.seriesdata.SeriesData
        Optical spectrum.
        float: array data is constructed as monochromatic spectrum.
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
        spectrum=(constants.speed_of_light / 780e-9), waist=100e-6,
        rotation_x=0.0, rotation_y=0.0, rotation_z=0.0, rotation=None,
        offset_x=0.0, offset_y=0.0, offset_z=0.0, offset=None
    ):
        super().__init__(pkg_name="libics-dev", cls_name="GaussianBeam")
        self.spectrum = spectrum
        self.waist = waist
        if rotation is None:
            rotation = np.array([rotation_x, rotation_y, rotation_z])
        self.rotation = np.array(rotation)
        if offset is None:
            offset = np.array([offset_x, offset_y, offset_z])
        self.offset = np.array(offset)

    @property
    def spectrum(self):
        return self._spectrum

    @spectrum.setter
    def spectrum(self, val):
        if isinstance(val, arraydata.ArrayData):
            self._spectrum = val
        elif isinstance(val, seriesdata.SeriesData):
            self._spectrum = cv_seriesdata_to_arraydata(val)
        elif np.isscalar(val):
            self._spectrum = arraydata.ArrayData()
            self._spectrum.add_dim(offset=val, name="frequency",
                                   symbol=r"\nu", unit="Hz")
            self._spectrum.add_dim(name="spectral density",
                                   symbol="S_A", unit="1/nm")
            self._spectrum.data = np.array([1])
        else:
            raise ValueError("invalid spectrum")

    @property
    def center_frequency(self):
        freqs = np.linspace(
            self._spectrum.scale.offset[0], self._spectrum.scale.max[0],
            num=self._spectrum.data.shape[0], endpoint=False
        )
        weights = self._spectrum.data / np.sum(self._spectrum.data)
        return np.sum(freqs * weights)

    @property
    def vacuum_wavenumber(self):
        return 2 * np.pi * self.center_frequency / constants.speed_of_light

    @property
    def vacuum_wavelength(self):
        return constants.speed_of_light / self.center_frequency

    @property
    def rayleigh_range(self):
        return np.pi * self.waist / self.vacuum_wavelength

    @property
    def rotation(self):
        return self._rotation

    @rotation.setter
    def rotation(self, ra):
        self._rotation = ra
        self._rotation_matrix = np.matmul(np.matmul(
            np.array([
                [1, 0, 0],
                [0, np.cos(ra[0]), -np.sin(ra[0])],
                [0, np.sin(ra[0]), np.cos(ra[0])]
            ]),
            np.array([
                [np.cos(ra[1]), 0, np.sin(ra[1])],
                [0, 1, 0],
                [-np.sin(ra[1]), 0, np.cos(ra[1])]
            ])
        ), np.array([
            [np.cos(ra[2]), -np.sin(ra[2]), 0],
            [np.sin(ra[2]), np.cos(ra[2]), 0],
            [0, 0, 1]
        ]))

    def cv_coordinates(self, x, y, z):
        """
        Performs inverse coordinate system rotation and translation
        as specified by the offset and rotation attributes.
        """
        var = np.array([x, y, z])
        # Numpy broadcasting support
        if len(var.shape) > 1:
            var = np.moveaxis(var, 0, -2)
            res = np.dot(self._rotation_matrix, var)
            res = np.moveaxis(np.moveaxis(res, 0, -1) - self.offset, -1, 0)
        else:
            res = np.dot(self._rotation_matrix, var) - self.offset
        return res

    def __call__(self, x, y, z=0.0, coord=COORD.CARTESIAN):
        """
        Calculates the Gaussian beam field with stored transformations.
        """
        k = self.vacuum_wavenumber
        w0 = self.waist
        zR = self.rayleigh_range

        if coord == COORD.POLAR:
            x, y = x * np.cos(y), x * np.sin(y)
        z = np.full_like(x, z)

        xx, yy, zz = self.cv_coordinates(x, y, z)
        r = np.sqrt(xx**2 + yy**2)
        wn = np.sqrt(1 + (zz / zR)**2)
        R = np.piecewise(
            zz, [zz == 0, zz != 0],
            [np.inf, lambda var: var + zR**2 / var]
        )
        res = (
            1 / wn * np.exp(-(r / w0 / wn)**2)
            * np.exp(-1j * k * r**2 / 2 / R)
            * np.exp(-1j * (k * zz + np.arctan(zz / zR)))
        )
        return res.astype(complex)
