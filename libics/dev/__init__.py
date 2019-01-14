from . import sim   # noqa


###############################################################################

import copy

import numpy as np
from scipy import signal, interpolate

from libics.data import types, arraydata, seriesdata
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
        cam_cfg=None, piezo_cfg=None, uncertainty=0.02,
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
        self.uncertainty = uncertainty

    def calc_coherence(self):
        """
        Calculates temporal and spatial coherence function.
        """
        mask = self.get_mask()
        mean = self.im_fixed.astype(float) + self.im_scanned.astype(float)
        norm = 2 * np.sqrt(self.im_fixed.astype(float)
                           * self.im_scanned.astype(float))
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

    def get_mask(self):
        """
        Generates a boolean area of same shape as the image by requiring a
        value larger than the parameter limit.
        """
        return (self.im_fixed.astype(float) * self.im_scanned.astype(float)
                > 1 / 4 / self.uncertainty**2)

    def _get_empty_image(self):
        """
        Gets an arraydata object only with camera image metadata.
        """
        im = arraydata.ArrayData()
        x_scale = self.cam_cfg.pixel_hrzt_size * 1e6
        y_scale = self.cam_cfg.pixel_vert_size * 1e6
        x_offset = self.cam_cfg.pixel_hrzt_offset * x_scale
        y_offset = self.cam_cfg.pixel_vert_offset * y_scale
        im.add_dim(offset=x_offset, scale=x_scale,
                   name="position", symbol="x", unit="µm")
        im.add_dim(offset=y_offset, scale=y_scale,
                   name="position", symbol="y", unit="µm")
        return im

    def get_image(self, im_name, mask=True):
        """
        Gets a stored array with metadata.

        Parameters
        ----------
        im_name : str
            "max", "min":
                Maximum, minimum of interfering image intensity.
            "fixed", "scanned":
                Fixed, scanned reference image.
            "mean":
                Incoherent superposition of fixed and scanned image.
            "max_voltage", "min_voltage":
                Piezo voltage corresponding to maximum, minimum image.
            "scoh":
                Spatial coherence.
        mask : bool
            Whether to mask image with np.nan.
            If True, also resizes the data to cut away all NaNs.

        Returns
        -------
        im : arraydata.ArrayData
            Requested image.

        Raises
        ------
        ValueError
            If im_name parameter is invalid.
        """
        im = self._get_empty_image()
        if im_name in ["max", "min", "fixed", "scanned"]:
            im.add_dim(name="intensity", symbol="I", unit="arb.")
            im.data = getattr(self, "im_" + im_name).astype(float)
        elif im_name == "mean":
            im.add_dim(name="intensity", symbol="I", unit="arb.")
            im.data = self.im_fixed.astype(float) + self.im_scanned
        elif im_name in ["max_voltage", "min_voltage"]:
            im.add_dim(name="piezo voltage", symbol="U", unit="V")
            im.data = getattr(self, "im_" + im_name[:3] + "_index")
        elif im_name == "scoh":
            im.add_dim(name="degree of coherence", symbol="|\gamma_A|")
            im.data = self.spatial_coherence
        else:
            raise ValueError("invalid im_name ({:s})".format(str(im_name)))
        if mask:
            if im_name in ["fixed", "scanned", "mean"]:
                im.data[im.data <= 5] = np.nan
            else:
                im.data[~self.get_mask()] = np.nan
            cut_min, cut_max = resize.resize_on_condition(
                im.data, cond="cut_all", val=np.nan
            )
            cut_slice = tuple([slice(cut_min[i], cut_max[i])
                               for i in range(len(cut_min))])
            for i, cmin in enumerate(cut_min):
                im.scale.offset[i] += im.scale.scale[i] * cmin
            im.data = im.data[cut_slice]
        return im

    def get_rescoh(self, scoh_factor=1):
        """
        Gets the statistics of the residual coherence.

        Parameters
        ----------
        scoh_factor : float or numpy.ndarray(2, float)
            Normalization factor for spatial coherence function
            (e.g. obtained from fitting). If the parameter is an
            array, it must have the same shape as the stored images.

        Returns
        -------
        rescoh : seriesdata.SeriesData
            Residual coherence as series data object.
            Has additional attributes for mean (mean) and
            standard deviation (std).
        """
        scoh = self.spatial_coherence.copy()
        if not isinstance(scoh_factor, np.ndarray):
            scoh_factor = np.full_like(scoh, scoh_factor, dtype=float)
        mask = ~np.logical_or(
            np.isnan(scoh), np.isnan(scoh_factor)
        )
        hist, bins = np.histogram(scoh[mask] / scoh_factor[mask], bins="auto")
        s_interm, s_cutoff, max_scoh = 0, hist.sum() * 0.95, bins[-1]
        for i, val in enumerate(hist):
            s_interm += val
            if s_interm > s_cutoff:
                max_scoh = bins[i]
                break
        scoh[scoh > max_scoh] = np.nan
        mask = ~np.logical_or(
            np.isnan(scoh), np.isnan(scoh_factor)
        )
        hist, bins = np.histogram(
            scoh[mask] / scoh_factor[mask],
            bins=int(300 * (np.nanmax(scoh[mask]) - np.nanmin(scoh[mask])))
        )

        bin_diff = bins[1:] - bins[:-1]
        bin_center = (bins[:-1] + bins[1:]) / 2
        pdf = hist / bin_diff
        pdf /= np.sum(pdf * bin_diff)

        rescoh = seriesdata.SeriesData()
        rescoh.add_dim(name="degree of coherence", symbol="\gamma_A")
        rescoh.add_dim(name="probability density", symbol="p")
        rescoh.data = np.array([bin_center, pdf])
        rescoh.mean = np.sum(bin_center * pdf * bin_diff)
        rescoh.std = np.sqrt(np.sum(
            (bin_center - rescoh.mean)**2 * pdf * bin_diff
        ))
        return rescoh


@InheritMap(map_key=("libics-dev", "InterferometerSequence"))
class InterferometerSequence(hdf.HDFBase):

    """
    items : list(InterferometerItem)
        List of interferometer items.
    displacements : np.array(1, float)
        List of lateral interferometer displacements.
    disp_quantity : data.types.Quantity
        Displacement quantity.
    params : list
        List of parameters corresponding to items.
    param_quantity : data.types.Quantity
        Parameter quantity.
    """

    def __init__(
        self, items=[], displacements=[],
        disp_quantity=types.Quantity(name="displacement", symbol="s"),
        params=[], param_quantity=types.Quantity()
    ):
        super().__init__(
            pkg_name="libics-dev", cls_name="InterferometerSequence"
        )
        self.items = misc.assume_list(items)
        self.displacements = np.array(displacements, dtype=float).flatten()
        self.disp_quantity = misc.assume_construct_obj(
            disp_quantity, types.Quantity
        )
        self.params = params
        self.param_quantity = misc.assume_construct_obj(
            param_quantity, types.Quantity
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
        disp = self.displacements
        weights = np.nanmean(50 * scoh, axis=(1, 2))
        # Logarithm of scaled Gaussian is parabola:
        # C = exp(c0 - c2 * x0^2), x0 = -c1 / 2 / c2, s = sqrt(-1 / 2 / c2)
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
        param[-3][param[-3] == 0] = np.nan
        self.scoh_offset = (-param[-2] / 2 / param[-3]).reshape(shape[1:])
        self.scoh_width = np.sqrt(-1 / param[-3] / 2).reshape(shape[1:])
        self.scoh_factor = np.exp(
            param[-1].reshape(shape[1:])
            - param[-3].reshape(shape[1:]) * self.scoh_offset**2
        )

    def get_mask(self):
        """
        Gets mutual mask between all interferometer items.
        """
        mask = self.items[0].get_mask()
        for item in self.items[1:]:
            mask = np.logical_and(mask, item.get_mask())
        return mask

    def get_image(self, im_name, mask=True):
        """
        Gets a stored array with metadata.

        Parameters
        ----------
        im_name : str
            "offset", "coherence_length", "factor".
        mask : bool
            Whether to mask image with np.nan.
            If True, also resizes the data to cut away all NaNs.

        Returns
        -------
        im : arraydata.ArrayData
            Requested image.

        Raises
        ------
        ValueError
            If im_name parameter is invalid.
        """
        im = self.items[0]._get_empty_image()
        if "offset" in im_name:
            im.add_dim(name="offset", symbol="s_0",
                       unit=self.disp_quantity.unit)
            im.data = self.scoh_offset
        elif "length" in im_name:
            im.add_dim(name="coherence length", symbol="s_c",
                       unit=self.disp_quantity.unit)
            im.data = self.scoh_width
        elif "factor" in im_name:
            im.add_dim(name="factor", symbol="C")
            im.data = self.scoh_factor
        else:
            raise ValueError("invalid im_name ({:s})".format(str(im_name)))
        if mask:
            im.data[~self.get_mask()] = np.nan
            cut_min, cut_max = resize.resize_on_condition(
                im.data, cond="cut_all", val=np.nan
            )
            cut_slice = tuple([slice(cut_min[i], cut_max[i])
                               for i in range(len(cut_min))])
            for i, cmin in enumerate(cut_min):
                im.scale.offset[i] += im.scale.scale[i] * cmin
            im.data = im.data[cut_slice]
        return im

    def get_scoh_function(self, index, coords):
        """
        Gets the spatial coherence function at a given index as callable.

        Parameters
        ----------
        index : tuple(int)
            Image index. If a slice is given, the function is averaged
            over the slice
        coords : np.ndarray(1, float)
            Displacement coordinates for evaluation.

        Returns
        -------
        scoh_data : np.ndarray(1, float)
            Interpolated spatial coherence function.
        scoh_fit : np.ndarray(1, float)
            Fitted spatial coherence function.
        """
        coords = np.array(coords)
        scoh_data = np.array([
            item.spatial_coherence[index] for item in self.items
        ])
        if scoh_data.ndim > 1:
            scoh_data = np.mean(
                scoh_data, axis=tuple(range(1, len(scoh_data.shape)))
            )
        scoh_data = interpolate.interp1d(
            self.displacements, scoh_data, kind="linear", fill_value=0
        )(coords)

        fit_factor = self.scoh_factor[index][..., np.newaxis]
        fit_offset = self.scoh_offset[index][..., np.newaxis]
        fit_width = self.scoh_width[index][..., np.newaxis]
        scoh_fit = np.mean(
            fit_factor * np.exp(-(coords - fit_offset)**2 / 2 / fit_width**2),
            axis=tuple(range(len(fit_factor.shape) - 1))
        )
        return scoh_data, scoh_fit

    def get_graph(self, *graph_names, sort=True):
        """
        Parameters
        ----------
        *graph_names : str
            "mean": Residual coherence mean.
            "std": Residual coherence standard deviation.
        sort : bool
            Whether to sort the parameters.

        Returns
        -------
        graph : seriesdata.SeriesData or list(SeriesData)
            Requested graph as a function of parameters.
            Returns list if multiple graph_names are given.
        """
        symbol = {
            "mean": "\overline{|\gamma_{A, \mathrm{res}}|}",
            "std": "\sigma_{|\gamma_{A, \mathrm{res}|}"
        }
        params = np.array(self.params)
        res = len(graph_names) * [None]
        graph = []
        rescoh = [item.get_rescoh() for item in self.items]
        for i, gn in enumerate(graph_names):
            res[i] = np.array([getattr(rc, gn) for rc in rescoh])
        if sort:
            index = np.argsort(params)
            params = params[index]
            for i in range(len(res)):
                res[i] = res[i][index]
        for i, gn in enumerate(graph_names):
            graph.append(seriesdata.SeriesData())
            graph[-1].add_dim(quantity=copy.deepcopy(self.param_quantity))
            graph[-1].add_dim(name="residual coherence " + gn,
                              symbol=symbol[gn])
            graph[-1].data = np.array([params, res[i]])
        if len(graph) == 1:
            graph = graph[0]
        return graph

    def get_rescoh(self):
        """
        Gets the statistics of the residual coherence of the interferometer
        furthest displaced from the maximum.

        Returns
        -------
        rescoh : seriesdata.SeriesData
            Residual coherence as series data object.
            Has additional attributes for mean (mean) and
            standard deviation (std).
        """
        disp_vals = self.displacements
        disp_min, disp_max = disp_vals.argmin(), disp_vals.argmax()
        index, offset = disp_min, self.scoh_offset.mean()
        if (np.abs(disp_vals[disp_max] - offset)
                > np.abs(disp_vals[disp_max] - offset)):
            index = disp_max
        return self.items[index].get_rescoh()

    def set_uncertainty(self, uncertainty):
        """
        Set relative uncertainty for all interferometer items.

        Parameters
        ----------
        uncertainty : float
            Allowed relative uncertainty of deduced coherence.
        """
        for item in self.items:
            item.uncertainty = uncertainty


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
