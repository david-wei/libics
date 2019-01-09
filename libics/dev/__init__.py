from . import sim   # noqa


###############################################################################


import numpy as np
from scipy import signal

from libics.file import hdf
from libics.util import InheritMap


###############################################################################


@InheritMap(map_key=("libics-dev", "InterferometerItem"))
class InterferometerItem(hdf.HDFBase):

    def __init__(
        self, voltages=None,
        im_max=None, im_max_index=None, im_min=None, im_min_index=None,
        im_fixed=None, im_scanned=None, trace=None, trace_coords=None,
        cam_cfg=None, piezo_cfg=None,
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
        self.trace_coords = trace_coords
        self.trace = trace
        self.cam_cfg = cam_cfg
        self.piezo_cfg = piezo_cfg

    def calc_coherence(self):
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
