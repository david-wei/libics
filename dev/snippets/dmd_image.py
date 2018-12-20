import sys
import time

import numpy as np
from scipy import interpolate, optimize

from PyQt5.QtCore import pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import (
    QApplication, QPushButton, QHBoxLayout, QVBoxLayout,
    QWidget, QFileDialog
)
# import pyqtgraph as pg

from libics.drv import drv, itf
from libics.util import misc, InheritMap
from libics.file import hdf
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

    def fit_peak_coordinates(self, image, snr=3):
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
        if image.max() / image.mean() < snr:
            return None
        xgrid, ygrid = np.meshgrid(
            np.arange(image.shape[0], np.arange(image.shape[1]))
        )
        xx, yy, zz = xgrid.flatten(), ygrid.flatten(), image.flatten()
        var = [(xx[i], yy[i]) for i in range(len(xx))]
        (_, x, y, _, _, _), _ = optimize.curve_fit(
            self.__fit_pc_func, var, zz
        )
        return x, y

    def __fit_at_func(self, vars, m11, m12, m21, m22, b1, b2):
        """
        Parameters
        ----------
        vars : cam_x, cam_y, dmd_x, dmd_y
        m_ij : matrix entries
        b_i : offset entries
        """
        return (
            (m11 * vars[0] + m12 * vars[1] + b1 - vars[2])**2
            + (m21 * vars[0] + m22 * vars[1] + b2 - vars[3])**2
        )

    def fit_affine_transform(self, cam_coords, dmd_coords):
        """
        Fits the affine transform matrix and offset vector.

        Parameters
        ----------
        cam_coords, dmd_coords : list(np.ndarray(1, float))
            List of (camera, DMD) coordinates in corresponding order.
        """
        vars = [np.concatenate(cam_coords[i], dmd_coords[i])
                for i in range(len(cam_coords))]
        res = np.full(len(cam_coords), 0, dtype=float)
        matrix = np.full((2, 2), np.nan, dtype=float)
        offset = np.full(2, np.nan, dtype=float)
        (matrix[0, 0], matrix[0, 1], matrix[1, 0], matrix[1, 1],
         offset[0], offset[1]), _ = optimize.curve_fit(
             self.__fit_at_func, vars, res
        )
        self.matrix = matrix
        self.offset = offset

    def calc_trafo(self, coordinates, images):
        """
        Estimates the transformation parameters.

        Parameters
        ----------
        coordinates : list((tuple(int)))
            List of DMD pixel coordinates.
        images : list(np.ndarray(2, float))
            List of camera images.

        Notes
        -----
        * Coordinate and image order must be identical.
        * At least two images are required. Additional data is used to obtain
          a more accurate map.
        """
        im_coords = []
        for i, (x, y) in enumerate(coordinates):
            im_coord = self.fit_peak_coordinates(images[i])
            if im_coord is not None:
                im_coords.append(im_coords)
        self.fit_affine_transform(im_coords, coordinates)

    # ++++ Perform transformation ++++

    def trafo(self, x, y):
        """
        Performs the affine transformation as specified by the offset,
        scale, rotation attributes.
        """
        var = np.array([x, y])
        # Numpy broadcasting support
        if len(var.shape) > 1:
            var = np.moveaxis(var, 0, -2)
            res = np.dot(self.matrix, var)
            res = np.moveaxis(res, 0, -1) + self.offset
            res = np.moveaxis(res, -1, 0)
        else:
            res = np.dot(self.matrix, var) + self.offset
        return res

    def inverse_trafo(self, x, y):
        """
        Performs the inverse affine transformation as specified by the offset,
        scale, rotation attributes.
        """
        var = np.array([x, y])
        # Numpy broadcasting support
        if len(var.shape) > 1:
            var = np.moveaxis(var, 0, -2)
            res = var - self.offset
            res = np.dot(np.linalg.inv(self.matrix), res)
            res = np.moveaxis(res, -1, 0)
        else:
            res = np.dot(np.linalg.inv(self.matrix), var - self.offset)
        return res

    def __call__(self, image, shape, mode="quintic"):
        return self.cv_cam_to_dsp(image, shape, mode=mode)

    def cv_cam_to_dsp(self, image, shape, mode="quintic"):
        """
        Convert camera image to DMD image.

        Parameter
        ---------
        image : np.ndarray(2, float)
            Beam profile image.
        shape : tuple(int)
            Shape of trafo_image.
        mode : "linear", "cubic", "quintic"
            Spline interpolation order.

        Returns
        -------
        trafo_image : np.ndarray(2, float)
            Beam profile on DMD.
        """
        xgrid, ygrid = np.meshgrid(
            np.arange(image.shape[0]), np.arange(image.shape[1]),
            indexing="ij"
        )
        xgrid, ygrid = self.trafo(xgrid, ygrid)
        f = interpolate.interp2d(xgrid, ygrid, image, kind=mode,
                                 bounds_error=False, fill_value=0)
        xnew, ynew = np.meshgrid(
            np.arange(shape[0]), np.arange(shape[1]), indexing="ij"
        )
        return f(xnew, ynew)

    def cv_dsp_to_cam(self, image, shape, mode="quintic"):
        """
        Convert DMD image to camera image.

        Parameter
        ---------
        image : np.ndarray(2, float)
            Beam profile image.
        shape : tuple(int)
            Shape of trafo_image.
        mode : "linear", "cubic", "quintic"
            Spline interpolation order.

        Returns
        -------
        trafo_image : np.ndarray(2, float)
            Beam profile on DMD.
        """
        xgrid, ygrid = np.meshgrid(
            np.arange(image.shape[0]), np.arange(image.shape[1]),
            indexing="ij"
        )
        xgrid, ygrid = self.inverse_trafo(xgrid, ygrid)
        f = interpolate.interp2d(xgrid, ygrid, image, kind=mode,
                                 bounds_error=False, fill_value=0)
        xnew, ynew = np.meshgrid(
            np.arange(shape[0]), np.arange(shape[1]), indexing="ij"
        )
        return f(xnew, ynew)


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
        self.pattern = None
        self.set_pattern(pattern="black")
        self.raw = np.full(im_resolution, np.nan, dtype=float)
        self.image = np.full(im_resolution, np.nan, dtype=float)
        self.rms = None
        self.save_data = None   # TODO: create save_data

    def setup(self):
        self.cam.setup()
        self.dsp.setup()

    def shutdown(self):
        self.cam.shutdown()
        self.dsp.shutdown()

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

    def find_trafo(self, coords=None, num=(11, 8)):
        """
        Uses a series of single pixel illuminations to estimate affine
        transformation between DMD and camera sensor.

        Parameters
        ----------
        coords : list((tuple(int)))
            List of DMD pixel coordinates to be used to
            transformation parameters.
            Overwrites the num parameter.
        num : int or tuple(int)
            Creates an equidistant grid of coords with
            num points. Tuples set the number of points
            in (horizontal, vertical) direction. Scalar
            num is interpreted as (num, num).
        """
        if coords is None:
            if isinstance(num, int):
                num = (num, num)
            x = np.linspace(0, self.dsp.cfg.pixel_hrzt_count.val,
                            num=num[0], dtype=int)
            y = np.linspace(0, self.dsp.cfg.pixel_vert_count.val,
                            num=num[1], dtype=int)
            xgrid, ygrid = np.meshgrid(x, y)
            xx, yy = xgrid.flatten()[np.newaxis], ygrid.flatten()[np.newaxis]
            coords = np.concatenate(xx, yy).T
        coords = np.array(coords, dtype=int)
        images = []
        for c in coords:
            dmd_image = np.full(
                (self.dsp.cfg.pixel_hrzt_count.val,
                 self.dsp.cfg.pixel_vert_count.val),
                0
            )
            dmd_image[tuple(c)] = 1
            self.dsp.init(dmd_image)
            self.dsp.write_all()
            self.dsp.run()
            time.sleep(0.1)
            im = self.cam.grab()
            images.append(np.copy(np.squeeze(im, axis=-1).T))
            self.dsp.stop()
        self.trafo.calc_trafo(coords, images)

    def load_trafo(self, file_path):
        """
        Loads an affine transformation file.

        Parameters
        ----------
        file_path : str
            File path to transformation file.
        """
        self.trafo = hdf.read_hdf(AffineTrafo, file_path=file_path)

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
                pattern = cv_error_diffusion(
                    calc_pattern(self.target, self.raw)
                )
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

    def load_image(self, file_path, record_raw=True):
        """
        Loads a target image from file and records a full on reference image.
        """
        # TODO: reset save_data
        if record_raw:
            self.record_raw()
        # load image

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
                self.target, self.image, param_p=0.7, param_s=0.1, cutoff=0.01
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
        # TODO: add to save_data

    def reset_save(self):
        """
        Resets the save_data.
        """
        self.save_data = None   # TODO:

    def save(self, file_path):
        """
        Saves recorded image, target image, reflectance pattern.
        """
        file_path = misc.assume_endswith(file_path, ".hdf5")
        hdf.write_hdf(self.save_data, file_path)


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
        self.qt_button_trafo_load = QPushButton("Load transformation")
        self.qt_button_trafo_find = QPushButton("Find transformation")
        self.qt_image_preview = qtimage.QtImage(aspect_ratio=1)
        self.qt_image_preview.set_image_format(channel="mono", bpc=8)
        self.qt_image_trafoview = qtimage.QtImage(aspect_ratio=1)
        self.qt_image_trafoview.set_image_format(channel="mono", bpc=8)
        self.qt_layout_preview.addWidget(self.qt_button_connect)
        self.qt_layout_preview.addWidget(self.qt_button_stop)
        self.qt_layout_preview.addWidget(self.qt_button_trafo_load)
        self.qt_layout_preview.addWidget(self.qt_button_trafo_find)
        self.qt_layout_preview.addWidget(self.qt_image_preview)
        self.qt_layout_preview.addWidget(self.qt_image_trafoview)

        self.qt_layout_meas = QVBoxLayout()
        self.qt_button_white = QPushButton("Set white pattern")
        self.qt_button_black = QPushButton("Set black pattern")
        self.qt_button_image_load = QPushButton("Load image")
        self.qt_button_image_iterate = QPushButton("Iterate image")
        self.qt_button_image_record = QPushButton("Record image")
        self.qt_button_save = QPushButton("Save data")
        self.qt_image_target = qtimage.QtImage(aspect_ratio=1)
        self.qt_image_target.set_image_format(channel="mono", bpc=8)
        self.qt_image_recorded = qtimage.QtImage(aspect_ratio=1)
        self.qt_image_recorded.set_image_format(channel="mono", bpc=8)
        self.qt_layout_meas.addWidget(self.qt_button_white)
        self.qt_layout_meas.addWidget(self.qt_button_black)
        self.qt_layout_meas.addWidget(self.qt_button_image_load)
        self.qt_layout_meas.addWidget(self.qt_button_image_iterate)
        self.qt_layout_meas.addWidget(self.qt_button_image_record)
        self.qt_layout_meas.addWidget(self.qt_button_save)
        self.qt_layout_meas.addWidget(self.qt_image_target)
        self.qt_layout_meas.addWidget(self.qt_image_recorded)

        self.qt_layout_main = QHBoxLayout()
        self.qt_layout_main.addLayout(self.qt_layout_preview)
        self.qt_layout_main.addLayout(self.qt_layout_meas)
        self.setWindowTitle("Digital Micromirror Device - Control")
        self.setLayout(self.qt_layout_main)

    def _init_visibility(self):
        super().show()

    def _init_connection(self):
        self.sUpdateImage.connect(self._on_update_image_emitted)
        self.qt_button_connect.clicked.connect(self._on_button_connect_clicked)
        self.qt_button_stop.clicked.connect(self._on_button_stop_clicked)
        self.qt_button_toggle.clicked.connect(self._on_button_toggle_clicked)
        self.qt_button_fixed.clicked.connect(self._on_button_fixed_clicked)
        self.qt_button_scanned.clicked.connect(self._on_button_scanned_clicked)
        self.qt_button_measure.clicked.connect(self._on_button_measure_clicked)
        self.qt_button_abort.clicked.connect(self._on_button_abort_clicked)
        self.qt_button_coherence.clicked.connect(
            self._on_button_coherence_clicked
        )
        self.qt_button_save.clicked.connect(self._on_button_save_clicked)
        self.qt_button_reset_piezo.clicked.connect(
            self._on_button_reset_piezo_clicked
        )

    @pyqtSlot(np.ndarray)
    def _on_update_image_emitted(self, im):
        self.qt_image_preview.update_image(im.astype("uint8"))

    @pyqtSlot()
    def _on_button_connect_clicked(self):
        self.connect()
        self.qt_button_connect.setVisible(False)
        self.qt_button_stop.setVisible(True)

    @pyqtSlot()
    def _on_button_stop_clicked(self):
        self.close()
        self.qt_button_connect.setVisible(True)
        self.qt_button_stop.setVisible(False)

    @pyqtSlot()
    def _on_button_save_clicked(self):
        file_path, _ = QFileDialog.getSaveFileName(
            caption="Save file as", filter="DMD save data (*.hdf5)"
        )
        if file_path == "":
            return
        self.save(file_path)


###############################################################################


if __name__ == "__main__":

    # Settings
    piezo_voltages = np.linspace(0, 75, num=501)
    piezo_address = "COM2"
    piezo_channel = "x"
    trace_coords = 3

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
    sys.exit(app_ret)
