from . import sim   # noqa


###############################################################################

import copy
import json
import os
import re

import numpy as np
from scipy import signal, interpolate, integrate

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
        mask = np.full_like(scoh, False, dtype=bool)
        mask[~np.isnan(scoh)] = mask[~np.isnan(scoh)] > max_scoh
        scoh[mask] = np.nan
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


# ++++++++++++++++++++++++++++


def _legacy_read_cohtrace(file_path):
    """
    Returns
    -------
    header : dict
        Keys: "piezo_trace", "cohtrace_coords", "crop_coords".
    cohtrace : np.ndarray(2, float)
        Array of temporal coherence traces.
    """
    lines = []
    with open(file_path) as f:
        for line in f:
            if re.match(r"#*", line):
                lines.append(line.strip("# \r\n"))
    header = None
    for line in lines:
        try:
            header = json.loads(line)
        except ValueError as e:
            continue
        if type(header) == dict:
            break
    assert(header is not None and
           "piezo_trace" in header.keys() and
           "cohtrace_coords" in header.keys() and
           "crop_coords" in header.keys())
    cohtrace = np.loadtxt(file_path)
    header["piezo_trace"] = cohtrace[0]
    cohtrace = cohtrace[1:]
    return header, cohtrace


def legacy_load_interferometer_item(
    folder, base_name=None, resolution=(1388, 1038), uncertainty=0.02
):
    """
    Legacy: Load .npy interferometer files.

    Parameters
    ----------
    folder : str
        Folder of stored files.
    base_name : str
        Base name of files. If None, uses the folder name.

    Returns
    -------
    intf : InterferometerItem
        Reconstructed interferometer item.
    """
    if base_name is None:
        base_name = os.path.basename(folder)
    prefix = os.path.join(folder, base_name)
    # Construct object
    intf = InterferometerItem(
        im_max=np.full(resolution, np.nan, dtype=float),
        im_max_index=np.zeros(resolution, dtype=int),
        im_min=np.full(resolution, np.nan, dtype=float),
        im_min_index=np.zeros(resolution, dtype=int),
        im_fixed=np.zeros(resolution, dtype=float),
        im_scanned=np.zeros(resolution, dtype=float),
        uncertainty=uncertainty
    )
    # Read metadata
    header, cohtrace = _legacy_read_cohtrace(prefix + "_cohtrace.txt")
    intf.voltages = header["piezo_trace"]
    intf.trace_coords = header["cohtrace_coords"]
    ((xmin, ymin), (xmax, ymax)) = header["crop_coords"]
    mask = (slice(xmin, xmax), slice(ymin, ymax))
    # Read images
    im_fixed = np.load(prefix + "_imref_fixed.npy")
    im_scanned = np.load(prefix + "_imref_scanned.npy")
    im_max = np.load(prefix + "_imrec_max.npy")
    im_min = np.load(prefix + "_imrec_min.npy")
    intf.im_max_index[mask] = np.load(prefix + "_impos_max.npy")
    intf.im_min_index[mask] = np.load(prefix + "_impos_min.npy")
    # Calculate back raw data
    i1 = 2 * np.sqrt(im_fixed.astype(float) * im_scanned)
    i2 = im_fixed.astype(float) + im_scanned
    intf.im_max[mask] = im_max * i1 + i2
    intf.im_min[mask] = im_min * i1 + i2
    intf.im_fixed[mask] = im_fixed
    intf.im_scanned[mask] = im_scanned
    trace = []
    for i, trace_item in enumerate(cohtrace):
        coord = tuple(intf.trace_coords[i])
        i1 = 2 * np.sqrt(intf.im_fixed[coord] * intf.im_scanned[coord])
        i2 = intf.im_fixed[coord] + intf.im_scanned[coord]
        trace.append(trace_item * i1 + i2)
    intf.trace = np.array(trace)
    return intf


###############################################################################


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
        # Interpolate NaN values
        nans, f = np.isnan(scoh), lambda z: z.nonzero()[0]
        scoh[nans] = np.interp(f(nans), f(~nans), scoh[~nans])
        # Perform fit
        weights = np.nanmean(20 * scoh, axis=(1, 2))
        scoh = np.log(scoh)
        poly_deg = 6 if len(disp) >= 10 else 2
        shape = scoh.shape
        param = np.polyfit(
            disp, scoh.reshape((shape[0], shape[1] * shape[2])),
            poly_deg, w=weights
        )
        self.scoh_params = param[::-1].reshape(
            (poly_deg + 1, shape[1], shape[2])
        )

        # Calculate offset and scale by maximum
        mask = self.get_mask()
        scoh_data = np.array([item.spatial_coherence for item in self.items],
                             dtype=float)[:, mask]
        scoh_index_max = np.nanargmax(scoh_data, axis=0)
        scoh_val_max = np.choose(scoh_index_max, scoh_data)
        self.scoh_offset = np.full_like(mask, fill_value=np.nan, dtype=float)
        self.scoh_scale = np.ones(mask.shape, dtype=float)
        self.scoh_offset[mask] = self.displacements[scoh_index_max]
        self.scoh_scale[mask] = scoh_val_max

        # Calculate coherence length
        mask = self.get_mask()
        self.scoh_length = np.full_like(mask, np.nan, dtype=float)
        scoh_scale = self.scoh_scale[mask][np.newaxis, ...]
        scoh_data = np.array([item.spatial_coherence for item in self.items],
                             dtype=float)[:, mask] / scoh_scale
        scoh_length = integrate.simps(scoh_data, self.displacements, axis=0)
        scoh_offset = self.scoh_offset[mask]
        ind_min = self.displacements.argmin()
        ind_max = self.displacements.argmax()
        disp_left = scoh_offset - self.displacements[ind_min]
        disp_right = scoh_offset + self.displacements[ind_max]
        scoh_length -= (disp_left * scoh_data[ind_min]**2
                        / 2 / np.log(scoh_data[ind_min]))
        scoh_length -= (disp_right * scoh_data[ind_max]**2
                        / 2 / np.log(scoh_data[ind_max]))
        self.scoh_length[mask] = scoh_length

    def get_mask(self):
        """
        Gets mutual mask between all interferometer items.
        """
        mask = self.items[0].get_mask()
        for item in self.items[1:]:
            mask = np.logical_and(mask, item.get_mask())
        return mask

    def get_mask_pos(self):
        """
        Gets the coordinates of the mask rectangle.

        Returns
        -------
        mask_pos : np.ndarray(2, int)
            Masking rectangle: ((x_min, y_min), (x_max, y_max)).
        """
        mask = self.get_mask()
        return resize.resize_on_condition(mask, cond="cut_all", val=False)

    def get_image(self, im_name, mask=True):
        """
        Gets a stored array with metadata.

        Parameters
        ----------
        im_name : str
            "scoh_length": Spatial coherence length.
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
        if "length" in im_name:
            im.add_dim(name="coherence length", symbol="s_c",
                       unit=self.disp_quantity.unit)
            im.data = self.scoh_length
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

    def get_scoh_function(self, index, coords, data_name="raw"):
        """
        Gets the spatial coherence function at a given index.

        Parameters
        ----------
        index : tuple(int)
            Image index. If a slice is given, the function is averaged
            over the slice
        coords : np.ndarray(1, float)
            Displacement coordinates for evaluation.
        data_name : str
            "raw": Actual measured data.
            "raw_corrected": Measured data shifted and scaled.
            "fit": Super-Gaussian fit data.
            "fit_corrected": Fitted data shifted and scaled.

        Returns
        -------
        sd : seriesdata.SeriesData
            Requested interpolated spatial coherence function.
        """
        coords = np.array(coords)
        coord_len = len(coords)
        # Spatial coherence data with shape (len(index), len(disp))
        scoh_data = np.array([item.spatial_coherence[index].flatten()
                              for item in self.items], dtype=float).T
        if scoh_data.ndim == 1:
            scoh_data = scoh_data[np.newaxis, ...]
        ind_len, disp_len = scoh_data.shape
        new_data = np.full((ind_len, coord_len), np.nan, dtype=float)
        # Calculate requested data
        if data_name == "raw":
            for i in range(ind_len):
                new_data[i] = interpolate.interp1d(
                    self.displacements, scoh_data[i],
                    kind="linear", bounds_error=False, fill_value=0
                )(coords)
        elif data_name == "raw_corrected":
            scoh_offset = self.scoh_offset[index].ravel()
            scoh_scale = self.scoh_scale[index].ravel()
            scoh_data /= scoh_scale[..., np.newaxis]
            for i, sco in enumerate(scoh_offset):
                new_data[i] = interpolate.interp1d(
                    self.displacements - sco, scoh_data[i],
                    kind="linear", bounds_error=False, fill_value=0
                )(coords)
        elif data_name == "fit":
            scoh_params = self.scoh_params[:, index[0], index[1]]
            scoh_params = scoh_params.reshape((len(scoh_params), ind_len)).T
            for i, param in enumerate(scoh_params):
                new_data[i] = np.exp(np.polynomial.polynomial
                                     .polyval(coords, param))
        elif data_name == "fit_corrected":
            scoh_params = self.scoh_params[:, index[0], index[1]]
            scoh_params = scoh_params.reshape((len(scoh_params), ind_len)).T
            scoh_offset = self.scoh_offset[index].ravel()
            scoh_scale = self.scoh_scale[index].ravel()
            for i, param in enumerate(scoh_params):
                new_data[i] = np.exp(
                    np.polynomial.polynomial
                    .polyval(coords + scoh_offset[i], param)
                ) / scoh_scale[i]
        else:
            raise ValueError("invalid data_name ({:s})".format(str(data_name)))
        scoh_data = np.nanmean(new_data, axis=0)
        # Result container
        sd = seriesdata.SeriesData()
        sd.add_dim(quantity=copy.deepcopy(self.disp_quantity))
        sd.add_dim(name="degree of coherence", symbol="|\gamma_A|")
        sd.data = np.array([coords, scoh_data])
        return sd

    def get_scoh_val(self, index, val_name="length"):
        """
        Gets a spatial coherence deduced value at a given index.

        Parameters
        ----------
        index : tuple(int)
            Image index. If a slice is given, the function is averaged
            over the slice
        val_name : str
            "length": Spatial coherence length.
            "offset": Coherence maximum offset.

        Returns
        -------
        scoh_val : data.types.ValQuantity
            Requested spatial coherence value.
        """
        names = {
            "length": "coherence length",
            "offset": "coherence centre"
        }
        symbols = {
            "length": "s_c",
            "offset": "s_0"
        }
        scv = None
        if val_name == "length":
            scv = np.nanmean(self.scoh_length[index])
        elif val_name == "offset":
            scv = np.nanmean(self.scoh_offset[index])
        scoh_val = types.ValQuantity(
            name=names[val_name], symbol=symbols[val_name],
            unit=self.disp_quantity.unit, val=scv
        )
        return scoh_val

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
        name = {
            "mean": "residual coherence mean",
            "std": "residual coherence standard deviation"
        }
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
            graph[-1].add_dim(name=name[gn], symbol=symbol[gn])
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
