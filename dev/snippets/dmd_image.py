import sys

import numpy as np
from scipy import interpolate

from PyQt5.QtCore import pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import (
    QApplication, QPushButton, QHBoxLayout, QVBoxLayout,
    QWidget, QFileDialog, QLabel
)
# import pyqtgraph as pg

from libics import dev
from libics.drv import drv, itf
from libics.util import misc, thread
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
        "protocol": itf.ITF_PROTOCOL.BINARY,
        "interface": itf.ITF_BIN.VIALUX,
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


class AffineTrafo(object):

    """
    Maps camera sensor pixel positions to DMD pixels.
    """

    def __init__(
        self,
        offset=np.array([0.0, 0.0]), scale=np.array([1.0, 1.0]), rotation=0
    ):
        self.offset = offset        # DMD side
        self.scale = scale          # Multiplication yields DMD side
        self.rotation = rotation

    def calc_peak_coordinates(self, image):
        """
        Uses a Gaussian fit to obtain the peak coordinates of an image.

        Parameters
        ----------
        image : np.ndarray(2, float)
            Image to be analyzed.

        Returns
        -------
        x, y : float, None
            (Fractional) image index coordinates of fit position.
            None: If fit failed.
        """

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
            im_coord = self.calc_peak_coordinates(images[i])
            if im_coord is not None:
                im_coords.append(im_coords)

    # ++++ Perform transformation ++++

    @property
    def rotation_matrix(self):
        return np.array([
            [np.cos(self.rotation), -np.sin(self.rotation)],
            [np.sin(self.rotation), np.cos(self.rotation)]
        ])

    @property
    def inverse_rotation_matrix(self):
        return np.array([
            [np.cos(self.rotation), np.sin(self.rotation)],
            [-np.sin(self.rotation), np.cos(self.rotation)]
        ])

    def trafo(self, x, y):
        """
        Performs the affine transformation as specified by the offset,
        scale, rotation attributes.
        """
        var = np.array([x, y])
        # Numpy broadcasting support
        if len(var.shape) > 1:
            var = np.moveaxis(var, 0, -2)
            res = np.dot(self.rotation_matrix, var)
            res = np.moveaxis(res, 0, -1) * self.scale + self.offset
            res = np.moveaxis(res, -1, 0)
        else:
            res = np.dot(self.rotation_matrix, var) * self.scale + self.offset
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
            res = var / self.scale - self.offset
            res = np.dot(self.inverse_rotation_matrix, var)
            res = np.moveaxis(res, -1, 0)
        else:
            res = np.dot(self.rotation_matrix, var / self.scale - self.offset)
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
    bool_pattern : np.ndarray(2, bool)
        Boolean image from error diffusion.
    """
    bool_pattern = np.zeros_like(target_pattern, dtype=bool)
    target_pattern = np.copy(target_pattern)
    for c1 in range(target_pattern.shape[0] - 1):
        for c2 in range(target_pattern.shape[1] - 1):
            if target_pattern[c1, c2] > 0.5:
                bool_pattern[c1, c2] = True
            else:
                bool_pattern[c1, c2] = False
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
    """

    def __init__(
        self, cam, dsp, im_callback=misc.do_nothing
    ):
        # Driver
        self.cam = cam
        self.dsp = dsp
        self.im_callback = im_callback
        im_resolution = np.array((
            cam.cfg.pixel_hrzt_count.val, cam.cfg.pixel_vert_count.val
        ))
        self.im_fixed = np.full(im_resolution, np.nan, dtype=float)

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

    def record_fixed(self):
        im = self.cam.grab()
        self.im_fixed = np.copy(np.squeeze(im, axis=-1).T)
        return im

    def record_trace(self, break_condition=None):
        """
        Parameters
        ----------
        break_condition : None or callable
            Function returning bool that is called to determine
            whether to break loop. Call signature: break_condition().
            If None, no break condition is set.
        """
        for volt in self.voltages:
            self.piezo.write_voltage(volt)
            im = np.squeeze(self.cam.grab(), axis=-1).T
            self.process_image(volt, im)
            if callable(break_condition):
                if break_condition():
                    break

    def process_image(self, voltage, image):
        self.im_max.add_element(voltage, image)
        self.im_min.add_element(voltage, image)
        self.trace.add_element(image)

    def calc_coherence(self):
        mask = np.logical_and(self.im_fixed > 5, self.im_scanned > 5)
        mean = self.im_fixed + self.im_scanned
        norm = 2 * np.sqrt(self.im_fixed * self.im_scanned)
        self.spatial_coherence = np.full_like(mean, np.nan)
        self.spatial_coherence[mask] = 0.5 * (
            np.abs((self.im_max.result[mask] - mean[mask]) / norm[mask])
            + np.abs((self.im_min.result[mask] - mean[mask]) / norm[mask])
        )
        mean = np.array([mean[tuple(coord)] for coord in self.trace.coords])
        norm = np.array([norm[tuple(coord)] for coord in self.trace.coords])
        trace_t = self.trace.trace.T
        self.temporal_coherence = ((trace_t - mean) / norm).T

    def save(self, file_path):
        file_path = misc.assume_endswith(file_path, ".hdf5")
        data = dev.InterferometerItem(
            self.voltages,
            self.im_max.result, self.im_max.index,
            self.im_min.result, self.im_min.index,
            self.im_fixed, self.im_scanned,
            self.trace.trace, self.trace.coords,
            self.cam.cfg, self.piezo.cfg
        )
        hdf.write_hdf(data, file_path)


###############################################################################


class DmdControlGui(DmdControl, QWidget):

    sUpdateImage = pyqtSignal(np.ndarray)
    sRecordTrace = pyqtSignal()
    sTraceRecorded = pyqtSignal()

    def __init__(
        self, cam=None, piezo=None, voltages=np.linspace(0, 75, num=501),
        trace_coords=3
    ):
        QWidget.__init__(self)
        if cam is None:
            cam = get_cam()
        if piezo is None:
            piezo = get_piezo()
        Interferometer.__init__(
            self,
            cam, piezo, voltages=voltages, trace_coords=trace_coords,
            im_callback=self.sUpdateImage.emit
        )
        self._init_gui()
        self._init_connection()
        self.setup()

    def __del__(self):
        self.shutdown()

    def _init_gui(self):
        self.qt_layout_preview = QVBoxLayout()
        self.qt_button_connect = QPushButton("Start camera")
        self.qt_button_stop = QPushButton("Stop camera")
        # self.qt_image_preview = pg.ImageView()
        self.qt_image_preview = qtimage.QtImage(aspect_ratio=1)
        self.qt_image_preview.set_image_format(channel="mono", bpc=8)
        self.qt_layout_preview.addWidget(self.qt_button_connect)
        self.qt_layout_preview.addWidget(self.qt_button_stop)
        self.qt_layout_preview.addWidget(self.qt_image_preview)

        self.qt_layout_ref = QVBoxLayout()
        self.qt_button_toggle = QPushButton("Toggle reference image")
        self.qt_button_fixed = QPushButton("Fixed image")
        self.qt_label_fixed = QLabel("Fixed image")
        # self.qt_image_fixed = pg.ImageView()
        self.qt_image_fixed = qtimage.QtImage(aspect_ratio=1)
        self.qt_image_fixed.set_image_format(channel="mono", bpc=8)
        self.qt_button_scanned = QPushButton("Scanned image")
        self.qt_label_scanned = QLabel("Scanned image")
        # self.qt_image_scanned = pg.ImageView()
        self.qt_image_scanned = qtimage.QtImage(aspect_ratio=1)
        self.qt_image_scanned.set_image_format(channel="mono", bpc=8)
        self.qt_layout_ref.addWidget(self.qt_button_toggle)
        self.qt_layout_ref.addWidget(self.qt_button_fixed)
        self.qt_layout_ref.addWidget(self.qt_button_scanned)
        self.qt_layout_ref.addWidget(self.qt_label_fixed)
        self.qt_layout_ref.addWidget(self.qt_label_scanned)
        self.qt_layout_ref.addWidget(self.qt_image_fixed)
        self.qt_layout_ref.addWidget(self.qt_image_scanned)

        self.qt_layout_coherence = QVBoxLayout()
        self.qt_button_measure = QPushButton("Measure coherence")
        self.qt_button_abort = QPushButton("Abort measurement")
        self.qt_button_coherence = QPushButton("Calculate coherence")
        self.qt_button_save = QPushButton("Save data")
        self.qt_button_reset_piezo = QPushButton("Reset piezo")
        # self.qt_image_coherence = pg.ImageView()
        self.qt_image_coherence = qtimage.QtImage(aspect_ratio=1)
        self.qt_image_coherence.set_image_format(channel="mono", bpc=8)
        self.qt_layout_coherence.addWidget(self.qt_button_measure)
        self.qt_layout_coherence.addWidget(self.qt_button_abort)
        self.qt_layout_coherence.addWidget(self.qt_button_coherence)
        self.qt_layout_coherence.addWidget(self.qt_button_save)
        self.qt_layout_coherence.addWidget(self.qt_button_reset_piezo)
        self.qt_layout_coherence.addWidget(self.qt_image_coherence)

        self.qt_layout_main = QHBoxLayout()
        self.qt_layout_main.addLayout(self.qt_layout_preview)
        self.qt_layout_main.addLayout(self.qt_layout_ref)
        self.qt_layout_main.addLayout(self.qt_layout_coherence)
        self.setWindowTitle("Interferometer - Coherence Measurement")
        self.setLayout(self.qt_layout_main)

    def _init_visibility(self):
        super().show()
        self.qt_button_connect.show()
        self.qt_button_stop.hide()
        self.qt_image_preview.show()
        self.qt_button_toggle.hide()
        self.qt_button_fixed.show()
        self.qt_label_fixed.hide()
        self.qt_image_fixed.hide()
        self.qt_button_scanned.show()
        self.qt_label_scanned.show()
        self.qt_image_scanned.show()
        self.qt_button_measure.hide()
        self.qt_button_abort.hide()
        self.qt_button_coherence.hide()
        self.qt_button_save.hide()
        self.qt_button_reset_piezo.show()
        self.qt_image_coherence.show()

    def _init_connection(self):
        self.sUpdateImage.connect(self._on_update_image_emitted)
        self.sRecordTrace.connect(self._on_record_trace_emitted)
        self.sTraceRecorded.connect(self._on_trace_recorded_emitted)
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
        # self.qt_image_preview.setImage(im)
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
    def _on_button_toggle_clicked(self):
        if self.qt_label_fixed.isVisible():
            self.qt_label_fixed.hide()
            self.qt_image_fixed.hide()
            self.qt_label_scanned.show()
            self.qt_image_scanned.show()
        elif self.qt_label_scanned.isVisible():
            self.qt_label_scanned.hide()
            self.qt_image_scanned.hide()
            self.qt_label_fixed.show()
            self.qt_image_fixed.show()

    @pyqtSlot()
    def _on_button_fixed_clicked(self):
        im = self.record_fixed()
        # self.qt_image_fixed.setImage(self.im_fixed)
        self.qt_image_fixed.update_image(im.astype("uint8"))
        self.qt_label_fixed.show()
        self.qt_image_fixed.show()
        self.qt_label_scanned.hide()
        self.qt_image_scanned.hide()
        if self._refs_set():
            self.qt_button_measure.show()
            self.qt_button_toggle.show()

    @pyqtSlot()
    def _on_button_scanned_clicked(self):
        im = self.record_scanned()
        # self.qt_image_scanned.setImage(self.im_scanned)
        self.qt_image_scanned.update_image(im.astype("uint8"))
        self.qt_label_scanned.show()
        self.qt_image_scanned.show()
        self.qt_label_fixed.hide()
        self.qt_image_fixed.hide()
        if self._refs_set():
            self.qt_button_measure.show()
            self.qt_button_toggle.show()

    @pyqtSlot()
    def _on_button_measure_clicked(self):
        self.sRecordTrace.emit()
        self.qt_button_measure.hide()
        self.qt_button_abort.show()
        self.qt_button_coherence.hide()
        self.qt_button_save.hide()

    def __record_trace_thread_function(self):
        self.record_trace(break_condition=(
            lambda: self.__record_trace_thread.stop_event.wait(timeout=0.0)
        ))
        self.sTraceRecorded.emit()

    @pyqtSlot()
    def _on_record_trace_emitted(self):
        self.__record_trace_thread = thread.StoppableThread()
        self.__record_trace_thread.run = self.__record_trace_thread_function
        self.__record_trace_thread.start()

    @pyqtSlot()
    def _on_button_abort_clicked(self):
        self.__record_trace_thread.stop()

    @pyqtSlot()
    def _on_trace_recorded_emitted(self):
        self.qt_button_measure.show()
        self.qt_button_abort.hide()
        self.qt_button_coherence.show()
        self.qt_button_save.show()

    @pyqtSlot()
    def _on_button_coherence_clicked(self):
        self.calc_coherence()
        # self.qt_image_coherence.setImage(self.spatial_coherence)
        coh_uint8 = (128 * self.spatial_coherence + 127).astype("uint8")
        coh_uint8 = np.expand_dims(coh_uint8, axis=0).T
        self.qt_image_coherence.update_image(coh_uint8)

    @pyqtSlot()
    def _on_button_save_clicked(self):
        file_path, _ = QFileDialog.getSaveFileName(
            caption="Save file as", filter="Coherence Items (*.hdf5)"
        )
        if file_path == "":
            return
        self.save(file_path)

    @pyqtSlot()
    def _on_button_reset_piezo_clicked(self):
        self.piezo.write_voltage(self.voltages[0])

    def _refs_set(self):
        return not (
            np.any(np.isnan(self.im_fixed))
            or np.any(np.isnan(self.im_scanned))
        )


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
