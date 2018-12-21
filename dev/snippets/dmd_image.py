import sys
import time

import PIL
import numpy as np
from scipy import ndimage, optimize

from PyQt5.QtCore import pyqtSignal, pyqtSlot
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QApplication, QPushButton, QHBoxLayout, QVBoxLayout,
    QWidget, QFileDialog
)
# import pyqtgraph as pg

from libics import env
from libics.drv import drv, itf
from libics.util import misc, InheritMap
from libics.file import hdf
from libics.trafo import resize
from libics.display import qtimage


###############################################################################


def get_cam(**kwargs):
    itf_cfg = {
        "protocol": itf.itf.ITF_PROTOCOL.BINARY,
        "interface": itf.itf.ITF_BIN.VIMBA,
        "device": None,     # first discovered camera
    }
    itf_cfg.update(kwargs)
    itf_cfg = itf.itf.ProtocolCfgBase(**itf_cfg).get_hl_cfg()
    drv_cfg = {
        "driver": drv.DRV_DRIVER.CAM,
        "interface": itf_cfg,
        "identifier": "alliedvision_manta_g145b_nir",
        "model": drv.DRV_MODEL.ALLIEDVISION_MANTA_G145B_NIR,
        "pixel_hrzt_count": 1388,
        "pixel_vert_count": 1038,
        "pixel_hrzt_size": 6.45e-6,
        "pixel_vert_size": 6.45e-6,
        "pixel_hrzt_offset": 0,
        "pixel_vert_offset": 0,
        "format_color": drv.DRV_CAM.FORMAT_COLOR.GS,
        "channel_bitdepth": 8,
        "exposure_mode": drv.DRV_CAM.EXPOSURE_MODE.MANUAL,
        "exposure_time": 1e-3,
        "acquisition_frames": 0,
        "sensitivity": drv.DRV_CAM.SENSITIVITY.NIR_HQ
    }
    drv_cfg.update(kwargs)
    drv_cfg = drv.DrvCfgBase(**drv_cfg).get_hl_cfg()
    cam = drv.DrvBase(cfg=drv_cfg).get_drv()
    return cam


def get_dsp(**kwargs):
    itf_cfg = {
        "protocol": itf.itf.ITF_PROTOCOL.BINARY,
        "interface": itf.itf.ITF_BIN.VIALUX,
        "device": None
    }
    itf_cfg.update(kwargs)
    itf_cfg = itf.itf.ProtocolCfgBase(**itf_cfg).get_hl_cfg()
    drv_cfg = {
        "driver": drv.DRV_DRIVER.DSP,
        "interface": itf_cfg,
        "identifier": "dmd_vialux_v7000_texasinstruments_dlp7000",
        "model": drv.DRV_MODEL.TEXASINSTRUMENTS_DLP7000,
        "pixel_hrzt_count": 1024,
        "pixel_hrzt_size": 13.68e-6,
        "pixel_hrzt_offset": 0,
        "pixel_vert_count": 768,
        "pixel_vert_size": 13.68e-6,
        "pixel_vert_offset": 0,
        "format_color": drv.DRV_DSP.FORMAT_COLOR.BW,
        "channel_bitdepth": 1,
        "picture_time": 9,
        "dark_time": 0,
        "sequence_repetitions": 0,
        "temperature": 25.0
    }
    drv_cfg.update(kwargs)
    drv_cfg = drv.DrvCfgBase(**drv_cfg).get_hl_cfg()
    dsp = drv.DrvBase(cfg=drv_cfg).get_drv()
    return dsp


###############################################################################


@InheritMap(map_key=("libics-dev", "AffineTrafo"))
class AffineTrafo(hdf.HDFBase):

    """
    Maps camera sensor pixel positions to DMD pixels.

    Convention: dmd_coord = matrix * cam_coord + offset.

    Usage
    -----
    * Take images for different single-pixel illuminations.
    * Calculate transformation parameters (calc_trafo method).
    * Perform transforms with call method
      (or cv_cam_to_dsp, cv_dsp_to_cam)
    """

    def __init__(
        self,
        matrix=np.diag([1, 1]).astype(float),
        offset=np.array([0, 0], dtype=float)
    ):
        super().__init__(pkg_name="libics-dev", cls_name="AffineTrafo")
        self.matrix = matrix
        self.offset = offset

    def __fit_pc_func(self, var, amp, cx, cy, sx, sy, off):
        """
        Parameters
        ----------
        var : (x, y) pixel indices
        amp : Gaussian amplitude
        cx, cy : (x, y) Gaussian center
        sx, sy : (x, y) Gaussian width
        off : Gaussian offset
        """
        exp = (var[0] - cx)**2 / sx**2 + (var[1] - cy)**2 / sy**2
        return amp * np.exp(-exp / 2) + off

    def fit_peak_coordinates(self, image, snr=1.5):
        """
        Uses a Gaussian fit to obtain the peak coordinates of an image.

        Parameters
        ----------
        image : np.ndarray(2, float)
            Image to be analyzed.
        snr : float
            Maximum-to-mean ratio required for fit.

        Returns
        -------
        x, y : float, None
            (Fractional) image index coordinates of fit position.
            None: If fit failed.
        """
        max_, mean = image.max(), image.mean()
        if max_ == 0 or mean == 0 or max_ / mean < snr:
            print("fit peak coordinates: insufficient snr")
            return None
        (xmin, ymin), (xmax, ymax) = resize.resize_on_mass(
            image, total_mass=0.5
        )
        xgrid, ygrid = np.meshgrid(
            np.arange(xmin, xmax), np.arange(ymin, ymax)
        )
        xx, yy = xgrid.flatten(), ygrid.flatten()
        zz = image[xmin:xmax, ymin:ymax].flatten()
        var = np.concatenate((xx[np.newaxis], yy[np.newaxis]))
        (_, x, y, _, _, _), cov = optimize.curve_fit(
            self.__fit_pc_func, var, zz
        )
        _, dx, dy, _, _, _ = np.diag(cov)
        if abs((dx - x) * x) > x**2 / 2 or abs((dy - y) * y) > y**2 / 2:
            print("fit peak coordinates: did not converge")
            return None
        return x, y

    def __lsq_at_func(self, param, var, func_val):
        """
        Parameters
        ----------
        param : np.ndarray(1, float)
            Array of parameters in the following order:
            m11, m12, m21, m22, b1, b2.
        var : np.ndarray(2, float)
            Independent variables with shape (2, N)
            as (x, y) camera coordinates.
            N is the number of data points.
        func_val : np.ndarray(2, float)
            Functional variables with shape (2, N)
            as (x, y) DMD coordinates.
        """
        chi2 = np.sum(
            (param[0] * var[0] + param[1] * var[1] + param[4] - func_val[0])**2
        )
        chi2 += np.sum(
            (param[2] * var[0] + param[3] * var[1] + param[5] - func_val[0])**2
        )
        return chi2

    def fit_affine_transform(
        self, cam_coords, dsp_coords, cam_shape, dsp_shape
    ):
        """
        Fits the affine transform matrix and offset vector.

        Parameters
        ----------
        cam_coords, dsp_coords : list(np.ndarray(1, float))
            List of (camera, DMD) coordinates in corresponding order.
        cam_shape, dsp_shape : tuple(int)
            (Camera, DMD) resolution.

        Returns
        -------
        success : bool
            Whether transformation fit succeeded.
            Matrix and offset attributes are only written
            in the case of success.
        """
        cam_res = np.array(cam_shape)
        dsp_res = np.array(dsp_shape)
        _angle = np.pi / 2
        _scale = np.mean(dsp_res / cam_res)
        matrix = _scale * np.array([
            [np.cos(_angle), np.sin(_angle)],
            [-np.sin(_angle), np.cos(_angle)]],
        )
        offset = dsp_res - cam_res
        cam_coords = np.array(cam_coords).T
        dsp_coords = np.array(dsp_coords).T
        res = optimize.least_squares(
            lambda x: self.__lsq_at_func(x, cam_coords, dsp_coords),
            (matrix[0, 0], matrix[0, 1], matrix[1, 0], matrix[1, 1],
             offset[0], offset[1]),
        )
        if res.success:
            (matrix[0, 0], matrix[0, 1], matrix[1, 0], matrix[1, 1],
             offset[0], offset[1]) = res.x
            self.matrix = matrix
            self.offset = offset
        return res.success

    def calc_trafo(self, cam_images, dsp_coords, dsp_shape=(1024, 768)):
        """
        Estimates the transformation parameters.

        Parameters
        ----------
        cam_images : list(np.ndarray(2, float))
            List of camera images.
        dsp_coords : list((tuple(int)))
            List of DMD pixel dsp_coords.
        dsp_shape : tuple(int)
            DMD resolution.

        Notes
        -----
        * Coordinate and image order must be identical.
        * At least two images are required. Additional data is used to obtain
          a more accurate map.
        """
        cam_coords = []
        ind_remove = []
        for i, (x, y) in enumerate(dsp_coords):
            print("\rFit peak dsp_coords {:d}".format(i), end="")
            im_coord = self.fit_peak_coordinates(cam_images[i])
            if im_coord is None:
                ind_remove.append(i)
            else:
                cam_coords.append(cam_coords)
        print("\nDelete coordinate items", ind_remove)
        dsp_coords = np.delete(dsp_coords, ind_remove, axis=0)
        if len(cam_coords) >= 2:
            print("Fit affine transform")
            self.fit_affine_transform(
                cam_coords, dsp_coords, cam_images[0].shape, dsp_shape
            )
        else:
            print("No valid images were taken")
            return None

    # ++++ Perform transformation ++++

    def __call__(self, image, shape, order=3, direction="cam_to_dsp"):
        if direction == "cam_to_dsp":
            return self.cv_cam_to_dsp(image, shape, order=order)
        elif direction == "dsp_to_cam":
            return self.cv_dsp_to_cam(image, shape, order=order)

    def cv_cam_to_dsp(self, image, shape, order=3):
        """
        Convert camera image to DMD image.

        Parameter
        ---------
        image : np.ndarray(2, float)
            Beam profile image.
        shape : tuple(int)
            Shape of trafo_image.
        order : int
            Spline interpolation order.

        Returns
        -------
        trafo_image : np.ndarray(2, float)
            Beam profile on DMD.
        """
        return ndimage.affine_transform(
            image, self.matrix, offset=self.offset, output_shape=shape,
            order=order, mode="constant", cval=0.0
        )

    def cv_dsp_to_cam(self, image, shape, order=3):
        """
        Convert DMD image to camera image.

        Parameter
        ---------
        image : np.ndarray(2, float)
            Beam profile image.
        shape : tuple(int)
            Shape of trafo_image.
        order : int
            Spline interpolation order.

        Returns
        -------
        trafo_image : np.ndarray(2, float)
            Beam profile on DMD.
        """
        inv_matrix = np.linalg.inv(self.matrix)
        inv_offset = -np.dot(inv_matrix, self.offset)
        return ndimage.affine_transform(
            image, inv_matrix, offset=inv_offset, output_shape=shape,
            order=order, mode="constant", cval=0.0
        )


###############################################################################


@InheritMap(map_key=("libics-dev", "DmdImageData"))
class DmdImageData(hdf.HDFBase):

    def __init__(self, target=None, trafo=None):
        super().__init__(pkg_name="libics-dev", cls_name="DmdImageData")
        self.target = target
        self.trafo = trafo
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


###############################################################################


def calc_pattern(target_image, raw_image):
    """
    Calculates the initial DMD reflectance pattern.

    Parameters
    ----------
    target_image : np.ndarray(2, float)
        Target beam profile image.
    raw_image : np.ndarray(2, float)
        Raw beam profile image for full DMD reflectance.

    Returns
    -------
    pattern : np.ndarray(2, float)
        Target DMD reflectance pattern.
    """
    pattern = target_image / raw_image
    pattern = pattern / np.max(pattern)
    return pattern


def iterate_pattern(
    target_image, actual_image,
    param_p=0.7, param_s=1.0, cutoff=0.1
):
    """
    Iterate DMD reflectance pattern using measured current output.

    Parameters
    ----------
    target_image : np.ndarray(2, float)
        Target beam profile on DMD.
    actual_image : np.ndarray(2, float)
        Actual beam profile on DMD.
    param_p, param_s : float
        Feedback parameters for:
        pattern = target + p * tanh((target - actual) / s)
    cutoff : float
        Relative pattern value below which reflectance
        is removed.

    Returns
    -------
    pattern : np.ndarray(2, float)
        Target reflectance pattern.
    """
    target_image = np.array(target_image, dtype=float)
    actual_image = np.array(actual_image, dtype=float)
    im_error = target_image - actual_image
    pattern = target_image + param_p * np.tanh(im_error / param_s)
    pattern = pattern / np.max(pattern)
    pattern[pattern < cutoff] = 0
    return pattern


def cv_error_diffusion(target_pattern):
    """
    Performs the error diffusion, transforming a greyscale to boolean image.

    Parameters
    ----------
    target_pattern : np.ndarray(2, float)
        Target greyscale image normalized to the interval [0, 1].

    Returns
    -------
    bool_pattern : np.ndarray(2, float)
        Boolean image (0., 1.) from error diffusion.
    """
    bool_pattern = np.zeros_like(target_pattern, dtype=float)
    target_pattern = np.copy(target_pattern)
    for c1 in range(target_pattern.shape[0] - 1):
        for c2 in range(target_pattern.shape[1] - 1):
            if target_pattern[c1, c2] > 0.5:
                bool_pattern[c1, c2] = 1
            else:
                bool_pattern[c1, c2] = 0
            err = target_pattern[c1, c2] - bool_pattern[c1, c2]
            target_pattern[c1, c2 + 1] += 7 / 16. * err
            target_pattern[c1 + 1, c2 + 1] += 1 / 16. * err
            target_pattern[c1 + 1, c2] += 5 / 16. * err
            target_pattern[c1 + 1, c2 - 1] += 3 / 16. * err
    return bool_pattern


def calc_deviation_rms(target_image, actual_image, mask=None):
    """
    Calculates the root-mean-square of the image ratio.

    Parameters
    ----------
    target_image : np.ndarray(2, float)
        Target beam profile image on DMD.
    actual_image : np.ndarray(2, float)
        Actual measured beam profile image on DMD.
    mask : None or float or np.ndarray(bool)
        np.ndarray:
            Boolean mask for which the rms deviation
            is calculated.
        None:
            No mask is applied.
        float:
            Calculation is masked by target values
            smaller than mask parameter.
    """
    if isinstance(mask, float):
        mask = (target_image > mask)
    if mask is not None:
        actual_image = actual_image[mask]
        target_image = target_image[mask]
    deviation = actual_image / target_image - 1
    rms = np.sqrt(deviation**2)
    return rms


###############################################################################


class DmdControl(object):

    """
    Parameters
    ----------
    cam : drv.drvcam.CamDrvBase
        Camera driver.
    dsp : drv.drvdsp.DspDrvBase
        DMD display driver.
    im_callback : callable
        Callback function on image recording.
        Call signature: im_callback(image)
        where image is a numpy.ndarray(2).

    Usage
    -----
    * Construction using camera and DMD drivers and optionally an
      on-image-ready callback function.
    * Initialize hardware using setup and connect methods.
    * To set a simple pattern, use set_pattern and
      display_pattern methods.
    * To perform greyscale image display:
        * Set the camera-to-DMD transformation parameters using
          find_trafo (measured) or load_trafo (from file)
          methods.
        * Load a target image using load_image method.
        * Use set_pattern to set the corresponding reflectance
          pattern and display_pattern to apply the pattern to
          hardware.
        * Use iterate_pattern to get correction feedback.
    * To analyze a resulting beam profile image, use the
      record_image method.
    * Use close and shutdown methods to end operation.
    """

    def __init__(
        self, cam, dsp, im_callback=misc.do_nothing
    ):
        self.cam = cam
        self.dsp = dsp
        self.im_callback = im_callback
        im_resolution = np.array((
            cam.cfg.pixel_hrzt_count.val, cam.cfg.pixel_vert_count.val
        ))
        self.trafo = AffineTrafo()
        self.pattern = np.zeros(im_resolution, dtype=float)
        self.raw = np.full(im_resolution, np.nan, dtype=float)
        self.image = np.full(im_resolution, np.nan, dtype=float)
        self.rms = None
        self.data = DmdImageData()

    def setup(self):
        self.cam.setup()
        self.dsp.setup()

    def shutdown(self):
        try:
            self.cam.shutdown()
        except AttributeError:
            pass
        try:
            self.dsp.shutdown()
        except AttributeError:
            pass

    def connect(self):
        self.cam.connect()
        self.dsp.connect()
        self.cam.read_all()
        self.dsp.read_all()
        self.cam.run(callback=self.im_callback)

    def close(self):
        self.cam.stop()
        self.cam.close()
        self.dsp.stop()
        self.dsp.close()

    # ++++++++++++++++++++++++++++++++

    @property
    def target_norm(self):
        """
        Returns target dsp image normalized to transformed raw image.
        """
        target_norm = self.target * self.raw.sum() / self.target.sum()
        return target_norm

    def find_cam_exposure(self, max_cutoff=2/3, max_iterations=50):
        """
        Varies the camera's exposure time to maximize image brightness
        while keeping the maximum ADC value below saturation.

        Parameters
        ----------
        max_cutoff : float
            Minimum relative brightness required for the image
            maximum.
        max_iterations : int
            Maximum number of iterations.

        Returns
        -------
        prev_exposure_time : float
            Previous exposure time in seconds (s).
            Allows the calling function to reset the settings.
        """
        prev_exposure_time = self.cam.cfg.exposure_time.val
        max_adc = float(2**self.cam.cfg.channel_bitdepth.val - 1)
        loop = True
        it = -1
        print("Finding exposure time")
        while loop:
            it += 1
            im = self.cam.grab()
            while im is None:
                im = self.cam.grab()
            saturation = im.max() / max_adc
            if saturation >= 1:
                self.cam.cfg.exposure_time.write(
                    val=self.cam.cfg.exposure_time.val * max_cutoff
                )
                self.cam.process()
            elif saturation < max_cutoff:
                self.cam.cfg.exposure_time.write(
                    val=self.cam.cfg.exposure_time.val / max_cutoff
                )
                self.cam.process()
            else:
                loop = False
            if it >= max_iterations:
                loop = False
            print("\r  {:.0f} µs (sat: {:.3f})       "
                  .format(self.cam.cfg.exposure_time.val * 1e6, saturation),
                  end="")
        print()
        return prev_exposure_time

    # ++++++++++++++++++++++++++++++++

    def find_trafo(self, coords=None, num=(11, 8), px=(5, 5)):
        """
        Uses a series of few-pixel illuminations to estimate affine
        transformation between DMD and camera sensor.

        Parameters
        ----------
        coords : list((tuple(int)))
            List of DMD pixel coordinates to be used to
            transformation parameters.
            Displayed pixels are
            [coords[i] - px[i] / 2 : coords[i] + px[i] / 2].
            If px[i] is uneven, the additional pixel is
            attached on the right interval side.
            Overwrites the num parameter.
        num : int or tuple(int)
            Creates an equidistant grid of coords with
            num points. Tuples set the number of points
            in (horizontal, vertical) direction. Scalar
            num is interpreted as (num, num).
        px : int or tuple(int)
            Rectangular (horizontal, vertical) number of
            pattern pixels used for one illumination.
            Minimum required px is 1.
            Scalar px is interpreted as (px, px).
        """
        res = (
            self.dsp.cfg.pixel_hrzt_count.val,
            self.dsp.cfg.pixel_vert_count.val
        )
        if isinstance(px, int):
            px = (px, px)
        dx = (px[0] - 1) // 2, px[0] - (px[0] - 1) // 2
        dy = (px[1] - 1) // 2, px[1] - (px[1] - 1) // 2
        if coords is None:
            if isinstance(num, int):
                num = (num, num)
            x = np.linspace(dx[0], res[0] - dx[1],
                            endpoint=True, num=num[0], dtype=int)
            y = np.linspace(dy[0], res[1] - dy[1],
                            endpoint=True, num=num[1], dtype=int)
            xgrid, ygrid = np.meshgrid(x, y)
            xx, yy = xgrid.flatten()[np.newaxis], ygrid.flatten()[np.newaxis]
            coords = np.concatenate((xx, yy)).T
        coords = np.array(coords, dtype=int)
        images = []
        prev_exposure_time = self.cam.cfg.exposure_time.val
        for c in coords:
            dmd_image = np.zeros(res, dtype=float)
            dmd_image[
                c[0] - dx[0]:c[0] + dx[1], c[1] - dy[0]:c[1] + dy[1]
            ] = 1
            self.set_pattern(pattern=dmd_image)
            self.display_pattern()
            time.sleep(2)
            self.find_cam_exposure()
            im = self.cam.grab()
            images.append(np.copy(np.squeeze(im, axis=-1).T))
            import matplotlib.pyplot as plt
            plt.subplot(121)
            plt.pcolormesh(dmd_image.T, vmin=0, vmax=1, cmap="jet")
            plt.gca().set_aspect(1)
            plt.subplot(122)
            plt.pcolormesh(images[-1].T, vmin=0, vmax=255)
            plt.gca().set_aspect(1)
            plt.colorbar()
            plt.show()
            self.dsp.stop()
        self.cam.cfg.exposure_time.write(val=prev_exposure_time)
        self.cam.process()
        self.trafo.calc_trafo(images, coords)
        self.data.trafo = self.trafo

    def load_trafo(self, file_path):
        """
        Loads an affine transformation file.

        Parameters
        ----------
        file_path : str
            File path to transformation file.
        """
        self.trafo = hdf.read_hdf(AffineTrafo, file_path=file_path)
        self.data.trafo = self.trafo

    # ++++++++++++++++++++++++++++++++

    def set_pattern(self, pattern="image"):
        """
        Sets a pattern to the DMD.

        Parameters
        ----------
        pattern : np.ndarray(2, float) or str or int
            Reflectance pattern.
            "image"
                Reflectance pattern according to loaded image
                using the error diffusion binarization.
            "white"
                Full on.
            "black"
                Full off.
            int
                Checkerboard with given number as linear
                square pixel count.
        """
        shape = (
            self.dsp.cfg.pixel_hrzt_count.val,
            self.dsp.cfg.pixel_vert_count.val
        )
        if isinstance(pattern, str):
            if pattern == "image":
                pattern = cv_error_diffusion(calc_pattern(
                    self.target_norm,
                    self.trafo(self.raw, self.target.shape)
                ))
            elif pattern == "white":
                pattern = np.ones(shape, dtype=float)
            elif pattern == "black":
                pattern = np.zeros(shape, dtype=float)
        elif isinstance(pattern, int):
            pattern = (pattern, pattern)
        if isinstance(pattern, tuple):
            pattern = np.kron(
                [[1, 0] * (shape[0] // pattern[0]),
                 [0, 1] * (shape[1] // pattern[1])],
                np.ones(pattern)
            )
            pattern = misc.resize_numpy_array(pattern, shape, fill_value=0)
        self.pattern = pattern

    def display_pattern(self):
        """
        Displays the loaded pattern.
        """
        self.dsp.stop()
        self.dsp.init(self.pattern)
        self.dsp.write_all()
        self.dsp.run()

    def record_raw(self):
        """
        Records an full on image.
        """
        self.set_pattern(pattern="white")
        self.display_pattern()
        time.sleep(0.1)
        self.raw = np.copy(np.squeeze(self.cam.grab(), axis=-1).T)
        self.image = np.copy(self.raw)
        self.data.raw = self.raw

    def load_image(self, file_path, record_raw=True):
        """
        Loads a target image from file and records a full on reference image.
        """
        self.data.reset()
        if record_raw:
            self.record_raw()
        im = np.array(PIL.Image.open(file_path).convert("L"))
        self.target = im.T
        self.data.target = self.target

    def iterate_pattern(self, num=1):
        """
        Sets the DMD reflectance pattern based on the loaded image and
        previously recorded white image.

        Parameters
        ----------
        num : int
            Number of iterations.

        Notes
        -----
        Assumes initial pattern was set and recorded.
        """
        for i in range(num):
            pattern = iterate_pattern(
                self.target_norm, self.trafo(self.image, self.target.shape),
                param_p=0.7, param_s=0.1, cutoff=0.01
            )
            self.set_pattern(pattern=pattern)
            self.record_image()

    def record_image(self):
        """
        Records and analyzes an image.
        """
        im = self.cam.grab()
        self.image = np.copy(np.squeeze(im, axis=-1).T)
        self.rms = calc_deviation_rms(self.target, self.image, mask=0.01)
        self.data.add_iteration(self.pattern, self.image, self.rms)

    def save_trafo(self, file_path):
        """
        Saves the transformation parameters.
        """
        file_path = misc.assume_endswith(file_path, ".hdf5")
        hdf.write_hdf(self.trafo, file_path=file_path)

    def save_data(self, file_path):
        """
        Saves recorded image, target image, reflectance pattern.
        """
        file_path = misc.assume_endswith(file_path, ".hdf5")
        hdf.write_hdf(self.data, file_path=file_path)


###############################################################################


class DmdControlGui(DmdControl, QWidget):

    sUpdateImage = pyqtSignal(np.ndarray)

    def __init__(
        self, cam=None, dsp=None
    ):
        QWidget.__init__(self)
        if cam is None:
            cam = get_cam()
        if dsp is None:
            dsp = get_dsp()
        DmdControl.__init__(
            self,
            cam, dsp, im_callback=self.sUpdateImage.emit
        )
        self._init_gui()
        self._init_connection()
        self.setup()

    def __del__(self):
        self.shutdown()

    def _init_gui(self):
        self.qt_layout_preview = QVBoxLayout()
        self.qt_button_connect = QPushButton("Start")
        self.qt_button_stop = QPushButton("Stop")
        self.qt_button_exposure_find = QPushButton("Find exposure time")
        self.qt_button_trafo_find = QPushButton("Find transformation")
        self.qt_button_trafo_load = QPushButton("Load transformation")
        self.qt_button_trafo_save = QPushButton("Save transformation")
        self.qt_image_preview = qtimage.QtImage(aspect_ratio=1)
        self.qt_image_preview.set_image_format(channel="mono", bpc=8)
        self.qt_image_pattern = qtimage.QtImage(aspect_ratio=1)
        self.qt_image_pattern.set_image_format(channel="mono", bpc=8)
        self.qt_layout_preview.addWidget(self.qt_button_connect)
        self.qt_layout_preview.addWidget(self.qt_button_stop)
        self.qt_layout_preview.addWidget(self.qt_button_exposure_find)
        self.qt_layout_preview.addWidget(self.qt_button_trafo_find)
        self.qt_layout_preview.addWidget(self.qt_button_trafo_load)
        self.qt_layout_preview.addWidget(self.qt_button_trafo_save)
        self.qt_layout_preview.addWidget(self.qt_image_preview)
        self.qt_layout_preview.addWidget(self.qt_image_pattern)

        self.qt_layout_meas = QVBoxLayout()
        self.qt_button_white = QPushButton("Set white pattern")
        self.qt_button_black = QPushButton("Set black pattern")
        self.qt_button_image_load = QPushButton("Load image")
        self.qt_button_image_iterate = QPushButton("Iterate image")
        self.qt_button_image_record = QPushButton("Record image")
        self.qt_button_data_save = QPushButton("Save data")
        self.qt_image_target = qtimage.QtImage(aspect_ratio=1)
        self.qt_image_target.set_image_format(channel="mono", bpc=8)
        self.qt_image_recorded = qtimage.QtImage(aspect_ratio=1)
        self.qt_image_recorded.set_image_format(channel="mono", bpc=8)
        self.qt_layout_meas.addWidget(self.qt_button_white)
        self.qt_layout_meas.addWidget(self.qt_button_black)
        self.qt_layout_meas.addWidget(self.qt_button_image_load)
        self.qt_layout_meas.addWidget(self.qt_button_image_iterate)
        self.qt_layout_meas.addWidget(self.qt_button_image_record)
        self.qt_layout_meas.addWidget(self.qt_button_data_save)
        self.qt_layout_meas.addWidget(self.qt_image_target)
        self.qt_layout_meas.addWidget(self.qt_image_recorded)

        self.qt_layout_main = QHBoxLayout()
        self.qt_layout_main.addLayout(self.qt_layout_preview)
        self.qt_layout_main.addLayout(self.qt_layout_meas)
        self.setWindowTitle("Digital Micromirror Device - Control")
        self.setLayout(self.qt_layout_main)

        self.setAutoFillBackground(True)
        bg_palette = self.palette()
        bg_color = env.colors.get_rgb_255("Greys", 9, 3)
        bg_palette.setColor(self.backgroundRole(), QColor(*bg_color))
        self.setPalette(bg_palette)

    def _init_visibility(self):
        super().show()
        self.qt_button_connect.show()
        self.qt_button_stop.hide()
        self.qt_button_exposure_find.show()
        self.qt_button_trafo_find.show()
        self.qt_button_trafo_load.show()
        self.qt_button_trafo_save.hide()
        self.qt_image_preview.show()
        self.qt_image_pattern.show()
        self.qt_button_white.show()
        self.qt_button_black.show()
        self.qt_button_image_load.show()
        self.qt_button_image_iterate.hide()
        self.qt_button_image_record.hide()
        self.qt_button_data_save.hide()
        self.qt_image_target.show()
        self.qt_image_recorded.show()

    def _init_connection(self):
        self.sUpdateImage.connect(self._on_update_image_emitted)

        self.qt_button_connect.clicked.connect(self._on_button_connect_clicked)
        self.qt_button_stop.clicked.connect(self._on_button_stop_clicked)
        self.qt_button_exposure_find.clicked.connect(self._on_button_exposure_find_clicked)
        self.qt_button_trafo_find.clicked.connect(
            self._on_button_trafo_find_clicked
        )
        self.qt_button_trafo_load.clicked.connect(
            self._on_button_trafo_load_clicked
        )
        self.qt_button_trafo_save.clicked.connect(
            self._on_button_trafo_save_clicked
        )

        self.qt_button_white.clicked.connect(self._on_button_white_clicked)
        self.qt_button_black.clicked.connect(self._on_button_black_clicked)
        self.qt_button_image_load.clicked.connect(
            self._on_button_image_load_clicked
        )
        self.qt_button_image_iterate.clicked.connect(
            self._on_button_image_iterate_clicked
        )
        self.qt_button_image_record.clicked.connect(
            self._on_button_image_record_clicked
        )
        self.qt_button_data_save.clicked.connect(
            self._on_button_data_save_clicked
        )

    def set_pattern(self, pattern="image"):
        super().set_pattern(pattern=pattern)
        im_pattern = (self.pattern * 255).astype("uint8").T
        self.qt_image_pattern.update_image(im_pattern[:, :, np.newaxis])
        self.display_pattern()

    @pyqtSlot(np.ndarray)
    def _on_update_image_emitted(self, im):
        self.qt_image_preview.update_image(im.astype("uint8"))

    @pyqtSlot()
    def _on_button_connect_clicked(self):
        self.connect()
        self.qt_button_connect.hide()
        self.qt_button_stop.show()

    @pyqtSlot()
    def _on_button_stop_clicked(self):
        self.close()
        self.qt_button_connect.show()
        self.qt_button_stop.hide()

    @pyqtSlot()
    def _on_button_exposure_find_clicked(self):
        self.find_cam_exposure()

    @pyqtSlot()
    def _on_button_trafo_find_clicked(self):
        self.find_trafo(num=5)  # TODO: DEBUG change num
        self.qt_button_trafo_save.show()

    @pyqtSlot()
    def _on_button_trafo_load_clicked(self):
        file_path, _ = QFileDialog.getOpenFileName(
            caption="Load transformation",
            filter="Affine transformation (*.hdf5)"
        )
        if file_path == "":
            return
        self.load_trafo(file_path)
        self.qt_button_trafo_save.show()

    @pyqtSlot()
    def _on_button_trafo_save_clicked(self):
        file_path, _ = QFileDialog.getSaveFileName(
            caption="Save file as", filter="Affine transformation (*.hdf5)"
        )
        if file_path == "":
            return
        self.save_trafo(file_path)

    @pyqtSlot()
    def _on_button_white_clicked(self):
        self.set_pattern(pattern="white")

    @pyqtSlot()
    def _on_button_black_clicked(self):
        self.set_pattern(pattern="black")

    @pyqtSlot()
    def _on_button_image_load_clicked(self):
        file_path, _ = QFileDialog.getOpenFileName(
            caption="Load image",
            filter="Bitmap (*.bmp), Portable Network Graphic (*.png)"
        )
        if file_path == "":
            return
        self.load_image(file_path)
        im_target = (self.target * 255).astype("uint8").T
        self.qt_image_target.update_image(im_target[:, :, np.newaxis])
        self.set_pattern(pattern="image")
        self.qt_button_image_iterate.show()
        self.qt_button_data_save.show()

    @pyqtSlot()
    def _on_button_image_iterate_clicked(self):
        self.iterate_pattern()

    @pyqtSlot()
    def _on_button_image_record_clicked(self):
        self.record_image()
        im_recorded = (self.image * 255).astype("uint8").T
        self.qt_image_recorded.update_image(im_recorded[:, :, np.newaxis])

    @pyqtSlot()
    def _on_button_data_save_clicked(self):
        file_path, _ = QFileDialog.getSaveFileName(
            caption="Save file as", filter="DMD image data (*.hdf5)"
        )
        if file_path == "":
            return
        self.save_data(file_path)


###############################################################################


if __name__ == "__main__":

    # Drivers
    cam = get_cam()
    dsp = get_dsp()

    # Run app
    app = QApplication(sys.argv)
    dmd_control = DmdControlGui(cam=cam, dsp=dsp)
    dmd_control._init_visibility()
    app_ret = app.exec_()

    # Clean up
    dmd_control.shutdown()
