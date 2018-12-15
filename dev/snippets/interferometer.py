# System Imports
import numpy as np
import sys

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


def get_piezo(**kwargs):
    itf_cfg = {
        "protocol": itf.itf.ITF_PROTOCOL.TEXT,
        "interface": itf.itf.ITF_TXT.SERIAL,
        "address": "COM2",
        "buffer_size": 1024,
        "send_timeout": 1.0,
        "send_termchar": "\r\n",
        "recv_timeout": 1.0,
        "recv_termchar": "\r\n",
        "baudrate": 115200,
        "bytesize": 8,
        "parity": "none",
        "stopbits": 1
    }
    itf_cfg.update(kwargs)
    itf_cfg = itf.itf.ProtocolCfgBase(**itf_cfg).get_hl_cfg()
    drv_cfg = {
        "driver": drv.DRV_DRIVER.PIEZO,
        "interface": itf_cfg,
        "identifier": "thorlabs_mdt693a",
        "model": drv.DRV_MODEL.THORLABS_MDT69XA,
        "limit_min": 0.0,
        "limit_max": 75.0,
        "displacement": 20e-6,
        "channel": "x",
        "feedback_mode": drv.DRV_PIEZO.FEEDBACK_MODE.OPEN_LOOP
    }
    drv_cfg.update(kwargs)
    drv_cfg = drv.DrvCfgBase(**drv_cfg).get_hl_cfg()
    piezo = drv.DrvBase(cfg=drv_cfg).get_drv()
    return piezo


###############################################################################


class ConditionalTrace(object):

    """
    Parameters
    ----------
    condition : callable
        Condition function when to replace results.
        Call signature: condition(new_element, old_element)
    shape : tuple(int)
        Shape of data.
    dtype : str
        Data type of result array.
    """

    def __init__(self, condition, shape, dtype=float):
        self.result = np.full(shape, np.nan, dtype=dtype)
        self.index = np.full(shape, np.nan)
        self.condition = condition

    def chk_condition(self, element):
        return np.logical_or(
            np.isnan(self.result),
            self.condition(element, self.result)
        )

    def add_element(self, index, element):
        mask = self.chk_condition(element)
        self.result[mask] = element[mask]
        self.index[mask] = index


class RecordedTrace(object):

    """
    Parameters
    ----------
    coords : list
        List describing trace, filters passed elements.
    length : int
        Length of trace.
    dtype : str
        Data type of trace.
    """

    def __init__(self, coords, length, dtype=float):
        self.coords = coords
        self.trace = np.full((len(coords), length), np.nan, dtype=dtype)
        self.__current_index = -1

    def add_element(self, element):
        self.__current_index += 1
        element = np.array([element[tuple(coords)] for coords in self.coords])
        self.trace[:, self.__current_index] = element


###############################################################################


class Interferometer(object):

    """
    Parameters
    ----------
    cam : drv.drvcam.CamDrvBase
        Camera driver.
    piezo : drv.drvpiezo.PiezoDrvBase
        Piezo driver.
    voltages : numpy.ndarray(1)
        Trace of piezo voltages.
    trace_coords : list(tuple(int)) or int
        List of image indices for which a trace should be recorded.
        If int, records central int x int pixels.
    im_callback : callable
        Callback function on image recording.
        Call signature: im_callback(image)
        where image is a numpy.ndarray(2).
    """

    def __init__(
        self, cam, piezo, voltages=np.linspace(0, 75, 501),
        trace_coords=3, im_callback=misc.do_nothing
    ):
        # Driver
        self.cam = cam
        self.piezo = piezo
        self.voltages = voltages
        self.im_callback = im_callback
        im_resolution = np.array((
            cam.cfg.pixel_hrzt_count.val, cam.cfg.pixel_vert_count.val
        ))
        # Coherence
        self.im_fixed = np.full(im_resolution, np.nan, dtype=float)
        self.im_scanned = np.full(im_resolution, np.nan, dtype=float)
        # Spatial coherence
        self.im_max = ConditionalTrace(np.greater, im_resolution, dtype=float)
        self.im_min = ConditionalTrace(np.less, im_resolution, dtype=float)
        # Temporal coherence
        if isinstance(trace_coords, int):
            center_coord = im_resolution // 2
            trace_coords = np.arange(trace_coords) - trace_coords // 2
            trace_coords = misc.get_combinations([trace_coords, trace_coords])
            trace_coords = [center_coord + np.array(it)
                            for it in trace_coords]
        self.trace = RecordedTrace(trace_coords, len(voltages), dtype=float)

    def setup(self):
        self.cam.setup()
        self.piezo.setup()

    def shutdown(self):
        self.cam.shutdown()
        self.piezo.shutdown()

    def connect(self):
        self.cam.connect()
        self.piezo.connect()
        self.cam.read_all()
        self.piezo.read_all()
        self.cam.run(callback=self.im_callback)

    def close(self):
        self.cam.stop()
        self.cam.close()
        self.piezo.close()

    def record_fixed(self):
        im = self.cam.grab()
        self.im_fixed = np.copy(np.squeeze(im, axis=-1).T)
        return im

    def record_scanned(self):
        im = self.cam.grab()
        self.im_scanned = np.copy(np.squeeze(im, axis=-1).T)
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


class InterferometerGui(Interferometer, QWidget):

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
    piezo = get_piezo(channel=piezo_channel, address=piezo_address)

    # Run app
    app = QApplication(sys.argv)
    interferometer = InterferometerGui(
        cam=cam, piezo=piezo, voltages=piezo_voltages,
        trace_coords=trace_coords
    )
    interferometer._init_visibility()
    app_ret = app.exec_()

    # Clean up
    interferometer.shutdown()
    sys.exit(app_ret)
