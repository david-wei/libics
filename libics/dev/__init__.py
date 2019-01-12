from . import sim   # noqa


###############################################################################

import copy

import numpy as np
from scipy import signal

from libics.data import types, arraydata
from libics.file import hdf
from libics.trafo import resize
from libics.util import InheritMap, misc


###############################################################################


@InheritMap(map_key=("libics-dev", "InterferometerItem"))
class InterferometerItem(hdf.HDFBase):

    def __init__(
        self, voltages=None,
        im_max=None, im_max_index=None, im_min=None, im_min_index=None,
        im_fixed=None, im_scanned=None, trace=None, trace_coords=None,
        cam_cfg=None, piezo_cfg=None, im_scale=None,
        pkg_name="libics-dev", cls_name="InterferometerItem"
    ):
        super().__init__(pkg_name=pkg_name, cls_name=cls_name)
        self.voltages = voltages
        self.im_max = im_max
        self.im_max_index = im_max_index
        self.im_min = im_min
        self.im_min_index = im_min_index
        self.im_fixed = im_fixed
        self.im_scanned = im_scanned
        self.im_scale = im_scale
        self.trace_coords = trace_coords
        self.trace = trace
        self.cam_cfg = cam_cfg
        self.piezo_cfg = piezo_cfg

    def calc_beam_pos(self):
        """
        Calculates the beam position and size for fixed and scanned images.
        """
        self.beam_pos_fixed = resize.resize_on_filter_maximum(
            self.im_fixed, min_val=0.05
        )
        self.beam_pos_scanned = resize.resize_on_filter_maximum(
            self.im_scanned, min_val=0.05
        )
        self.beam_pos_mean = (
            (max(self.beam_pos_fixed[0][0], self.beam_pos_scanned[0][0]),
             max(self.beam_pos_fixed[0][1], self.beam_pos_scanned[0][1])),
            (min(self.beam_pos_fixed[1][0], self.beam_pos_scanned[1][0]),
             min(self.beam_pos_fixed[1][1], self.beam_pos_scanned[1][1]))
        )

    def calc_coherence(self):
        """
        Calculates temporal and spatial coherence function.
        """
        mask = np.logical_and(self.im_fixed > 5, self.im_scanned > 5)
        mean = self.im_fixed.astype(float) + self.im_scanned.astype(float)
        norm = (2 * np.sqrt(self.im_fixed.astype(float)
                * self.im_scanned.astype(float)))
        self.spatial_coherence = np.full_like(mean, np.nan, dtype=float)
        self.spatial_coherence[mask] = 0.5 * (
            np.abs((self.im_max[mask] - mean[mask]) / norm[mask])
            + np.abs((self.im_min[mask] - mean[mask]) / norm[mask])
        )
        mean = np.array([mean[tuple(coord)] for coord in self.trace_coords])
        norm = np.array([norm[tuple(coord)] for coord in self.trace_coords])
        trace_t = self.trace.T
        self.temporal_coherence = ((trace_t - mean) / norm).T

    def _find_position(self, bin_index, trace_length):
        dist_index = np.full(len(bin_index) + 1, np.nan)
        dist_index[1:-1] = 1 / (bin_index[1:] - bin_index[:-1])
        dist_index[0], dist_index[-1] = dist_index[1], dist_index[-2]
        spacing = []
        counter = 0
        for it, index in enumerate(bin_index):
            while(counter < index):
                spacing.append(dist_index[it])
                counter += 1
        while(counter < trace_length):
            spacing.append(dist_index[-1])
            counter += 1
        previous = 0
        position = []
        for sp in spacing:
            previous += sp
            position.append(previous)
        return np.array(position)

    def calc_calibration(self, wavelength=780e-9):
        position_estimate = []
        for tc in self.temporal_coherence:
            index_max = signal.find_peaks(tc)[0]
            index_min = signal.find_peaks(-tc)[0]
            if index_max.size > 10:
                position_estimate.append(
                    self._find_position(index_max, len(tc)) * wavelength / 2
                )
                position_estimate.append(
                    self._find_position(index_min, len(tc)) * wavelength / 2
                )
        self.position_estimate = np.array(position_estimate).mean(axis=0)

    def calc_speckle_contrast(self):
        pass

    def get_rescoh(self):
        """
        Gets a histogram of the residual coherence.

        Returns
        -------
        bins : np.ndarray(1)
            Bin centers of histogram.
        hist : np.ndarray(1)
            Histogram values.
        mean : float
            Mean value.
        std : float
            Standard deviation.
        """
        scoh = self.spatial_coherence
        mask = np.logical_not(np.logical_or(
            np.isnan(scoh), np.isnan(self.scoh_factor)
        ))
        hist, bins = np.histogram(scoh[mask] / self.scoh_factor[mask])
        bins = (bins[:-1] + bins[1:]) / 2
        mean = bins * hist / hist.sum()
        std = (bins - mean)**2 * hist / hist.sum()
        return bins, hist, mean, std


@InheritMap(map_key=("libics-dev", "InterferometerSequence"))
class InterferometerSequence(hdf.HDFBase):

    """
    items : list(InterferometerItem)
        List of interferometer items.
    displacements : list(data.types.ValQuantity)
        List of lateral interferometer displacements.
    im_scale : data.arraydata.ArrayScale
        Metadata for camera image.
    piezo_calibr : trafo.data.Calibration
        Calibration of piezo voltage to longitudinal position.
    """

    def __init__(self,
                 items=[], displacements=[], im_scale=None, piezo_calibr=None):
        super().__init__(
            pkg_name="libics-dev", cls_name="InterferometerSequence"
        )
        self.items = misc.assume_list(items)
        self.displacements = misc.assume_list(displacements)
        for i, item in enumerate(self.displacements):
            self.displacements[i] = misc.assume_construct_obj(
                item, types.ValQuantity
            )
        self.im_scale = misc.assume_construct_obj(
            im_scale, arraydata.ArrayScale
        )

    def calc_coherence(self):
        for item in self.items:
            item.calc_coherence()

    def calc_beam_pos(self):
        for item in self.items:
            item.calc_beam_pos()

    def calc_speckle_contrast(self):
        for item in self.items:
            item.calc_speckle_contrast()

    def calc_spatial_coherence_scale(self):
        """
        Estimates the displacement offset and scaling factor for each pixel.
        """
        scoh = np.array([item.spatial_coherence for item in self.items])
        disp = np.array([item.val for item in self.displacements])
        weights = np.nanmean(scoh, axis=(1, 2))
        # Logarithm of scaled Gaussian is parabola:
        # C = exp(c0 - x0^2), x0 = -c1 / 2 / c2, s = sqrt(c2 / 2)
        scoh = np.log(scoh)
        # Interpolate NaN values
        nans, f = np.isnan(scoh), lambda z: z.nonzero()[0]
        scoh[nans] = np.interp(f(nans), f(~nans), scoh[~nans])
        # Perform fit
        poly_deg = 2
        shape = scoh.shape
        param = np.polyfit(
            disp, scoh.reshape((shape[0], shape[1] * shape[2])),
            poly_deg, w=weights
        )
        param[2][param[2] == 0] = np.nan
        self.scoh_offset = (-param[1] / 2 / param[2]).reshape(shape[1:])
        self.scoh_width = np.sqrt(-param[2] / 2).reshape(shape[1:])
        self.scoh_factor = np.exp(
            param[0].reshape(shape[1:]) - self.scoh_offset**2
        )

    def get_scoh(self):
        """
        Gets the spatially resolved spatial coherence properties.

        Returns
        -------
        offset, coherence_length, factor : arraydata.ArrayData
            Gaussian fit parameters.
        """
        scoh = []
        symbol = ["s_0", "s_c", "C"]
        for i, name in enumerate(["offset", "coherence length", "factor"]):
            sc = arraydata.ArrayData()
            sc.scale = copy.deepcopy(self.im_scale)
            sc.scale.quantity[-1] = types.Quantity(
                name=name, symbol=symbol[i], unit=self.displacements[0].unit
            )
            if name == "offset":
                sc.data = self.scoh_offset
            elif name == "coherence length":
                sc.data = self.scoh_width
            if name == "factor":
                sc.data = self.scoh_factor
            scoh.append(sc)
        return scoh

    def get_rescoh(self):
        """
        Gets a histogram of the residual coherence.

        Returns
        -------
        bins : np.ndarray(1)
            Bin centers of histogram.
        hist : np.ndarray(1)
            Histogram values.
        mean : float
            Mean value.
        std : float
            Standard deviation.
        """
        disp_vals = np.array([item.val for item in self.displacements])
        disp_min, disp_max = disp_vals.argmin(), disp_vals.argmax()
        index, offset = disp_min, self.scoh_offset.mean()
        if (np.abs(disp_vals[disp_max] - offset)
                > np.abs(disp_vals[disp_max] - offset)):
            index = disp_max
        return self.items[index].get_rescoh()


###############################################################################


@InheritMap(map_key=("libics-dev", "DmdImageData"))
class DmdImageData(hdf.HDFBase):

    def __init__(self, target=None, trafo=None, raw=None):
        super().__init__(pkg_name="libics-dev", cls_name="DmdImageData")
        self.target = target
        self.trafo = trafo
        self.raw = raw
        self.patterns = []
        self.images = []
        self.rms = []

    def reset(self):
        self.patterns = []
        self.images = []
        self.rms = []

    def add_iteration(self, pattern, image, rms):
        self.patterns.append(pattern)
        self.images.append(image)
        self.rms.append(rms)
