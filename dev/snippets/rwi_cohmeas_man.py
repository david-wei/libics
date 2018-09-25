# System Imports
from collections import deque
import json
import numpy as np
import os
import sys
import threading
import time

# Qt Imports
from PyQt5.QtCore import pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import (
    QApplication, QPushButton, QHBoxLayout, QVBoxLayout, QLabel,
    QSizePolicy, QWidget, QFileDialog, QMessageBox
)
import pyqtgraph as pg

# Package Imports
from libics.cfg import err as ERR
from libics.cfg import env as ENV
from libics.display import qtimage
from libics.drv import cam, piezo
from libics.drv.itf import camcfg
from libics.util.thread import PeriodicTimer
from libics.trafo import resize


###############################################################################


def get_camera_cfg(
    camera_type="vimba", camera_id=None,
    color_sensitivity="normal",
    width=1388, height=1038, width_offset=0, height_offset=0,
    exposure_time=None
):
    camera_cfg = camcfg.CameraCfg()

    camera_cfg.camera.camera_type.set_val(camera_type, diff_flag=False)
    camera_cfg.camera.camera_id.set_val(camera_id, diff_flag=False)

    camera_cfg.image_format.width.set_val(width)
    camera_cfg.image_format.height.set_val(height)
    camera_cfg.image_format.width_offset.set_val(width_offset)
    camera_cfg.image_format.height_offset.set_val(height_offset)
    camera_cfg.image_format.channel.set_val("mono")
    camera_cfg.image_format.bpc.set_val(8)

    camera_cfg.acquisition.frame_count.set_val(1)
    camera_cfg.acquisition.color_sensitivity.set_val(color_sensitivity)

    camera_cfg.exposure.cam_auto.set_val("off")
    if exposure_time is None:
        camera_cfg.exposure.time = None
    else:
        camera_cfg.exposure.time.set_val(exposure_time)

    return camera_cfg


def get_piezo_cfg(
    piezo_type="mdt693", piezo_id=0, port=None, min_volt=0.0, max_volt=75.0
):
    piezo_cfg = piezo.PiezoCfg()

    piezo_cfg.device.device_type.set_val(piezo_type, diff_flag=False)
    piezo_cfg.device.device_id.set_val(piezo_id, diff_flag=False)
    piezo_cfg.device.port.set_val(port, diff_flag=False)

    piezo_cfg.voltage.voltage_min.set_val(min_volt)
    piezo_cfg.voltage.voltage_max.set_val(max_volt)

    return piezo_cfg


###############################################################################


class CohMeas(QWidget, object):

    """
    Provides a Qt widget to measure spatial and temporal coherence with a
    piezo transducer controlled white-light interferometer.

    Parameters
    ----------
    camera_cfg : drv.itf.camcfg.CameraCfg
        Camera configuration for recording coherence traces.
    piezo_cfg : drv.piezo.PiezoCfg
        Piezo (driver) configuration.
    step_time : float
        Waiting time in seconds between piezo steps.
    piezo_steps : int
        Number of steps the piezo scan is divided into.
    piezo_trace : str or list(float)
        `str`:
            `"bilinear"`, `"bilinear_reversed"`, `"linear_up"`,
            `"linear_down"`
        `list(float)`:
            List of voltages. `piezo_steps` parameter is ignored.
    cohtraces : int
        Linear dimension number of previewed coherence traces.
    *args, **kwargs
        Passed to constructor of `QWidget`.

    Constants
    ---------
    MODES : PREVIEW, NORMVIEW, REFFIXED, REFSCANNED
        Possible recording modes (`NORMVIEW` is measurement mode).
    """

    PREVIEW = 0
    NORMVIEW = 1
    REFFIXED = 2
    REFSCANNED = 3

    sUpdatePlot = pyqtSignal()

    def __init__(self,
                 camera_cfg, piezo_cfg, *args,
                 step_time=1e-1, piezo_steps=1000, piezo_trace="bilinear",
                 cohtraces=3, image_buffer=200, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_camera(camera_cfg)
        self._init_piezo(piezo_cfg)
        self._init_logic(step_time, piezo_steps, piezo_trace,
                         cohtraces, image_buffer)
        self._init_qt_preview()
        self._init_qt_normview()
        self._init_qt_cohtrace()
        self._init_ui()
        self._init_connections()

    def shutdown(self):
        """
        Closes all interface connections.
        """
        # self.piezo.close_piezo()
        # self.piezo.shutdown_piezo()
        self.camera.stop()
        self.camera.close_camera()
        self.camera.shutdown_camera()

    def _init_logic(self, step_time, piezo_steps, piezo_trace,
                    cohtraces, image_buffer):
        # Access control
        self.mode = CohMeas.PREVIEW
        self._cam_access = threading.Lock()
        self._cohtrace_access = threading.Lock()
        self._measurement_timer = None
        # Normalization images
        self._image_buffer_count = image_buffer
        self._ref_fixed_image = None
        self._ref_scanned_image = None
        self._ref_norm_image = None
        self._ref_mean_image = None
        self._crop_coords = ((0, 0), (self.camera_cfg.image_format.width.val,
                                      self.camera_cfg.image_format.height.val))
        self._normalized_images = deque(maxlen=self._image_buffer_count)
        self._image_records = {"max": None, "min": None, "sum": None}
        self._image_pos = {"max": None, "min": None}
        self._image_record_counter = 0
        # Scanning settings
        self._step_time = step_time
        self._piezo_trace = []
        self._piezo_trace_counter = -1
        self.set_piezo_trace(piezo_trace=piezo_trace, piezo_steps=piezo_steps)
        self._cohtrace_coords = []
        self._update_cohtrace_coords(cohtraces)

    def _init_camera(self, camera_cfg):
        self.camera_cfg = camera_cfg
        self.camera = cam.Camera(camera_cfg=camcfg.CameraCfg(
            camera_type=self.camera_cfg.camera.camera_type.val,
            camera_id=self.camera_cfg.camera.camera_id.val
        ))
        self.camera.set_origin_callback(self.process_callback)
        self.camera.disable_frame_buffer()
        self.camera.setup_camera()
        self.camera.open_camera()
        self.camera.read_camera_cfg(overwrite_cfg=True)
        self.camera.set_camera_cfg(self.camera_cfg)
        self.camera.write_camera_cfg()

    def _init_piezo(self, piezo_cfg):
        self.piezo_cfg = piezo_cfg
        # self.piezo = piezo.Piezo(piezo_cfg=piezo.PiezoCfg(
        #     device_type=self.piezo_cfg.device.device_type.val,
        #     device_id=self.piezo_cfg.device.device_id.val,
        #     port=self.piezo_cfg.device.port.val
        # ))
        # self.piezo.setup_piezo()
        # self.piezo.open_piezo()
        # self.piezo.read_piezo_cfg(overwrite_cfg=True)
        # self.piezo.set_piezo_cfg(self.piezo_cfg)
        # self.piezo.write_piezo_cfg()

    # ++++ GUI ++++++++++++++++++++++++++++++++++++

    def _init_qt_preview(self):
        self.qt_preview = qtimage.QtImage(aspect_ratio=1)
        self.qt_preview.set_image_format(
            channel=self.camera_cfg.image_format.channel.val,
            bpc=self.camera_cfg.image_format.bpc.val
        )

    def _init_qt_normview(self):
        self.qt_normview = qtimage.QtImage(aspect_ratio=1)
        self.qt_normview.set_image_format(channel=None, bpc=8)

    def _init_qt_cohtrace(self):
        self.qt_cohtrace = pg.GraphicsLayoutWidget(parent=self)
        self.qt_cohtrace.setBackground(None)
        self._cohtrace_data = []
        self._cohtrace_plots = []
        for it in range(len(self._cohtrace_coords)):
            self._cohtrace_data.append([])
            self._cohtrace_plots.append(self.qt_cohtrace
                                        .addPlot(row=it, col=0))
            self._cohtrace_plots[-1].setLabels(
                left=("Pos ({:d}, {:d})"
                      .format(*self._cohtrace_coords[it])),
                bottom="Piezo voltage [V]"
            )
            self._cohtrace_plots[-1].showGrid(x=True, y=True)
        self._update_cohtrace_plot_range()

    def _init_ui(self):
        # Main window
        self.setWindowTitle(
            "Reversing Wavefront Interferometer - Coherence Measurement"
        )
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        # Main layout
        self.controls_layout = QHBoxLayout()
        self.display_layout = QHBoxLayout()
        self.main_layout.addLayout(self.controls_layout)
        self.main_layout.addLayout(self.display_layout)

        # Display layout
        self.preview_layout = QVBoxLayout()
        self.preview_layout.setSpacing(0)
        self.preview_layout.setContentsMargins(0, 0, 0, 0)
        self.cohtrace_layout = QVBoxLayout()
        self.display_layout.addLayout(self.preview_layout, stretch=1)
        self.display_layout.addLayout(self.cohtrace_layout, stretch=2)

        # Controls Layout
        self._button_live_preview = QPushButton("Live preview")
        self._button_live_preview.setCheckable(True)
        self._button_record_ref_fixed = QPushButton(
            "Record fixed arm image"
        )
        self._button_record_ref_scanned = QPushButton(
            "Record scanned arm image"
        )
        self._button_measure_coherence = QPushButton("Measure coherence")
        self._button_measure_coherence.setCheckable(True)
        self._button_measure_coherence.setEnabled(False)
        self._button_save_data = QPushButton("Save data")
        self.controls_layout.addWidget(self._button_live_preview)
        self.controls_layout.addWidget(self._button_record_ref_fixed)
        self.controls_layout.addWidget(self._button_record_ref_scanned)
        self.controls_layout.addWidget(self._button_measure_coherence)
        self.controls_layout.addWidget(self._button_save_data)

        # Preview layout
        self._label_preview = QLabel("Raw image")
        self._label_preview.setSizePolicy(
            QSizePolicy.Minimum, QSizePolicy.Fixed
        )
        self._label_normview = QLabel("Normalized image")
        self._label_normview.setSizePolicy(
            QSizePolicy.Minimum, QSizePolicy.Fixed
        )
        self.preview_layout.addWidget(self._label_preview)
        self.preview_layout.addWidget(self.qt_preview)
        self.preview_layout.addWidget(self._label_normview)
        self.preview_layout.addWidget(self.qt_normview)

        # Cohtrace layout
        self._label_cohtrace = QLabel("Normalized interference intensity")
        self._label_cohtrace.setSizePolicy(
            QSizePolicy.Minimum, QSizePolicy.Fixed
        )
        self.cohtrace_layout.addWidget(self._label_cohtrace)
        self.cohtrace_layout.addWidget(self.qt_cohtrace)

    def _init_connections(self):
        self._button_live_preview.toggled.connect(self.toggle_live_preview)
        self._button_record_ref_fixed.clicked.connect(
            self.record_image_ref_fixed
        )
        self._button_record_ref_scanned.clicked.connect(
            self.record_image_ref_scanned
        )
        self._button_measure_coherence.toggled.connect(
            self.toggle_measure_coherence
        )
        self._button_save_data.clicked.connect(self.save_data)
        self.sUpdatePlot.connect(self._update_cohtrace_plots)

    def _uncheck_button_measure_coherence(self):
        self._button_measure_coherence.setChecked(False)

    def show(self):
        super().show()
        self.qt_preview.show()
        self.qt_normview.show()
        self.qt_cohtrace.show()

    @pyqtSlot(bool)
    def toggle_live_preview(self, is_checked):
        if is_checked:
            self._button_measure_coherence.setChecked(False)
            time.sleep(ENV.THREAD_DELAY_QTSIGNAL)
        self._cam_access.acquire()
        self.camera.stop()
        self._set_camera_continuous(continuous=is_checked)
        if is_checked:
            self.mode = CohMeas.PREVIEW
            self.camera.run()
        self._cam_access.release()

    @pyqtSlot()
    def record_image_ref_fixed(self):
        is_live_previewing = self._button_live_preview.isChecked()
        self._button_live_preview.setChecked(False)
        self._button_measure_coherence.setChecked(False)
        time.sleep(ENV.THREAD_DELAY_QTSIGNAL)
        self._cam_access.acquire()
        self.camera.stop()
        self.mode = CohMeas.REFFIXED
        self.camera.run()
        self._cam_access.release()
        self._button_live_preview.setChecked(is_live_previewing)

    @pyqtSlot()
    def record_image_ref_scanned(self):
        is_live_previewing = self._button_live_preview.isChecked()
        self._button_live_preview.setChecked(False)
        self._button_measure_coherence.setChecked(False)
        time.sleep(ENV.THREAD_DELAY_QTSIGNAL)
        self._cam_access.acquire()
        self.camera.stop()
        self.mode = CohMeas.REFSCANNED
        self.camera.run()
        self._cam_access.release()
        self._button_live_preview.setChecked(is_live_previewing)

    @pyqtSlot(bool)
    def toggle_measure_coherence(self, is_checked):
        if not self._ref_are_set():
            if is_checked:
                self._button_measure_coherence.setChecked(False)
            return
        if is_checked:
            self._button_live_preview.setChecked(False)
            time.sleep(ENV.THREAD_DELAY_QTSIGNAL)
            self._cam_access.acquire()
            self._button_live_preview.setEnabled(False)
            self._button_record_ref_fixed.setEnabled(False)
            self._button_record_ref_scanned.setEnabled(False)
            self.mode = CohMeas.NORMVIEW
            self.run_measurement()
            self._cam_access.release()
        else:
            self.stop_measurement()
            self._button_live_preview.setEnabled(True)
            self._button_record_ref_fixed.setEnabled(True)
            self._button_record_ref_scanned.setEnabled(True)

    @pyqtSlot()
    def save_data(self):
        # Get save directory
        dialog = QFileDialog()
        save_dir = dialog.getExistingDirectory(
            caption="Choose directory in which data is saved"
        )
        if save_dir == "":
            return
        _head, tail_name = os.path.split(save_dir)
        if tail_name == "":
            _, tail_name = os.path.split(_head)
        # Save cohtraces
        header = {}
        header["piezo_trace"] = "V"
        header["cohtrace_coords"] = self._cohtrace_coords
        header = json.dumps(header)
        data = np.array([self._piezo_trace] + self._cohtrace_data)
        np.savetxt(
            os.path.join(save_dir, tail_name + "_cohtrace.txt"),
            data, header=header
        )
        # Save image records
        np.save(
            os.path.join(save_dir, tail_name + "_imrec_max.np"),
            self._image_records["max"],
            allow_pickle=False
        )
        np.save(
            os.path.join(save_dir, tail_name + "_impos_max.np"),
            self._image_pos["max"],
            allow_pickle=False
        )
        np.save(
            os.path.join(save_dir, tail_name + "_imrec_min.np"),
            self._image_records["min"],
            allow_pickle=False
        )
        np.save(
            os.path.join(save_dir, tail_name + "_impos_min.np"),
            self._image_pos["min"],
            allow_pickle=False
        )
        np.save(
            os.path.join(save_dir, tail_name + "_imrec_mean.np"),
            self._image_records["sum"] / self._image_record_counter,
            allow_pickle=False
        )

    # ++++ Logic functions ++++++++++++++++++++++++

    def _set_camera_continuous(self, continuous=True):
        """
        Changes the camera framecount mode between continuous and single
        frame mode.
        """
        self.camera.get_camera_cfg().acquisition.frame_count.set_val(
            0 if continuous else 1,
            diff_flag=True
        )
        self.camera.write_camera_cfg()

    def set_piezo_trace(self, piezo_trace="bilinear", piezo_steps=1000):
        """
        Sets the voltage trace the piezo will scan.

        Parameters
        ----------
        piezo_steps : int
            Number of steps the piezo scan is divided into.
        piezo_trace : str or list(float)
            `str`:
                `"bilinear"`, `"bilinear_reversed"`, `"linear_up"`,
                `"linear_down"`
            `list(float)`:
                List of voltages. `piezo_steps` parameter is ignored.

        Notes
        -----
        Updates the plot range to show the whole trace.
        """
        if type(piezo_trace) == str:
            start = self.piezo_cfg.voltage.voltage_min.val
            end = self.piezo_cfg.voltage.voltage_max.val
            if piezo_trace == "bilinear":
                num = int(round(piezo_steps / 2))
                self._piezo_trace = np.concatenate((
                    np.linspace(start, end, num=num),
                    np.linspace(end, start, num=num)
                ))
            elif piezo_trace == "bilinear_reversed":
                num = int(round(piezo_steps / 2))
                self._piezo_trace = np.concatenate((
                    np.linspace(end, start, num=num),
                    np.linspace(start, end, num=num)
                ))
            elif piezo_trace == "linear_up":
                self._piezo_trace = np.linspace(start, end, num=piezo_steps)
            elif piezo_trace == "linear_down":
                self._piezo_trace = np.linspace(end, start, num=piezo_steps)
            else:
                raise ERR.INVAL_SET(ERR.INVAL_SET.str("invalid piezo trace"))
        else:
            self._piezo_trace = np.array(piezo_trace)
        self._update_cohtrace_plot_range()

    def _update_cohtrace_plot_range(self):
        # check whether in initialization process
        if hasattr(self, "_cohtrace_plots") and hasattr(self, "_piezo_trace"):
            piezo_range = (min(self._piezo_trace), max(self._piezo_trace))
            for pl in self._cohtrace_plots:
                pl.setXRange(*piezo_range)
                pl.setYRange(-1.2, 1.2)

    @pyqtSlot()
    def _update_cohtrace_plots(self):
        self._cohtrace_access.acquire()
        for it in range(len(self._cohtrace_plots)):
            self._cohtrace_plots[it].plot(
                x=self._piezo_trace[: len(self._cohtrace_data[it])],
                y=self._cohtrace_data[it],
                clear=True
            )
        self._cohtrace_access.release()

    def _set_piezo_voltage(self, index=None, voltage=None):
        """
        Sets the piezo voltage.

        Parameters
        ----------
        index : int
            Gets the voltage stored in `piezo_trace` at `index`
            position. If `None`, chooses the next index after the
            current `piezo_trace_counter`.
        voltage : float
            If `not None`, sets the given voltage. Ignores the
            `index` parameter.
        """
        if voltage is None:
            if index is None:
                index = ((self._piezo_trace_counter + 1)
                         % len(self._piezo_trace))
            voltage = self._piezo_trace[index]
            self._piezo_trace_counter = index
        # self.piezo.set_voltage(voltage)
        msg_box = QMessageBox()
        msg_box.setWindowTitle("Set piezo voltage")
        msg_box.setText("Manually set piezo voltage to {:.2f}V.\n" +
                        "Press OK to continue, press cancel to measurement.")
        msg_box.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        ret = msg_box.exec_()
        return ret == QMessageBox.Ok

    def _ref_are_set(self):
        """
        Returns whether the reference images are loaded.
        """
        return (
            self._ref_fixed_image is not None and
            self._ref_scanned_image is not None
        )

    def _update_cohtrace_coords(self, cohtraces, update_plot_label=False):
        """
        Updates the cohtrace coordinates from the crop coordinates.

        Parameters
        ----------
        cohtraces : int
            Linear number of cohtraces to be recorded.
        update_plot_label : bool
            Whether to set qt_cohtraces axes labels.
            `False` only used at initialization.
        """
        d_width, d_height = (
            int(round((self._crop_coords[1, 0] - self._crop_coords[0, 0])
                      / (cohtraces + 1))),
            int(round((self._crop_coords[1, 1] - self._crop_coords[0, 1])
                      / (cohtraces + 1)))
        )
        self._cohtrace_coords = []
        for w in range(cohtraces):
            for h in range(cohtraces):
                self._cohtrace_coords.append((
                    max(0, (h + 1) * d_height - 1),
                    max(0, (w + 1) * d_width - 1)
                ))
        if update_plot_label:
            for it, plot in enumerate(self._cohtrace_plots):
                plot.setLabels(
                    left=("Pos ({:d}, {:d})"
                          .format(*self._cohtrace_coords[it])),
                    bottom="Piezo voltage [V]"
                )

    def _update_ref_images(self):
        """
        Updates the normalization images.
        """
        if self._ref_are_set():
            self._ref_mean_image = np.add(
                self._ref_fixed_image, self._ref_scanned_image,
                dtype="float64"
            )
            self._crop_coords = resize.resize_on_mass(
                self._ref_mean_image, center="auto", total=self._crop_mass,
                aspect_ratio="auto", aspect_mode="enlarge"
            )
            self._ref_norm_image = 2 * np.sqrt(np.multiply(
                self._ref_fixed_image, self._ref_scanned_image,
                dtype="float64"
            ))
            cohtraces = int(np.sqrt(len(self._cohtrace_coords)))
            self._update_cohtrace_coords(cohtraces, update_plot_label=True)
            self._button_measure_coherence.setEnabled(True)

    def _reset_image_records(self):
        """
        Resets the image records.
        """
        shape = (
            (self._crop_coords[1][0] - self._crop_coords[0][0]),
            (self._crop_coords[1][1] - self._crop_coords[0][1])
        )
        self._image_records["max"] = np.full(shape, 0.0, dtype="float64")
        self._image_records["min"] = np.full(shape, 0.0, dtype="float64")
        self._image_records["sum"] = np.full(shape, 0.0, dtype="float64")
        self._image_pos["max"] = np.full(shape, 0.0, dtype="float64")
        self._image_pos["min"] = np.full(shape, 0.0, dtype="float64")
        self._image_record_counter = 0

    def run_measurement(self):
        """
        Constructs a timer which periodically shifts the piezo voltage
        and acquires an image.
        """
        self._normalized_images = deque(maxlen=self._image_buffer_count)
        self._reset_image_records()
        for it in range(len(self._cohtrace_data)):
            self._cohtrace_data[it] = []
        self._measurement_timer = PeriodicTimer(
            self._step_time, self._measurement_timer_callback,
            repetitions=len(self._piezo_trace)
        )
        self._measurement_timer.set_stop_action(
            self._uncheck_button_measure_coherence
        )
        self._measurement_timer.start()

    def stop_measurement(self):
        """
        Stops the measurement timer.
        """
        self._measurement_timer.stop()
        self._piezo_trace_counter = -1

    # ++++ Callback functions +++++++++++++++++++++

    def _measurement_timer_callback(self):
        """
        Measurement timer callback function which stops and runs the camera in
        single image mode.
        """
        if not self._set_piezo_voltage():
            self._button_measure_coherence.setChecked(False)
        time.sleep(ENV.THREAD_DELAY_COM)
        self._cam_access.acquire()
        self.camera.stop()
        self.camera.run()
        self._cam_access.release()

    def process_callback(self, np_image):
        """
        Camera callback function which calls the processing methods associated
        with the current `mode`.
        """
        self.display_preview(np_image)
        # Eliminate rgb dimension
        if self.mode != CohMeas.PREVIEW:
            if len(np_image.shape) == 3:
                np_image = np.mean(np_image, 2)
            if self.mode == CohMeas.NORMVIEW:
                self.append_normview(np_image)
                self.display_normview()
                self.display_cohtrace()
            elif self.mode == CohMeas.REFFIXED:
                self.set_ref_image(np_image, "fixed")
            elif self.mode == CohMeas.REFSCANNED:
                self.set_ref_image(np_image, "scanned")

    def display_preview(self, np_image):
        """
        Updates the preview image.
        """
        self.qt_preview.update_image(np_image)

    def display_normview(self):
        """
        Updates the normview image.
        """
        normview = (128 * self._normalized_images[-1] + 127).astype("uint8")
        self.qt_normview.update_image(normview)

    def display_cohtrace(self):
        """
        Updates the cohtrace plots.
        """
        self.sUpdatePlot.emit()

    def append_normview(self, np_image):
        """
        Calculates and appends a normalized image.

        Parameters
        ----------
        np_image : numpy.ndarray(dtype=uint8)
            8-bit grayscale image from camera.

        Raises
        ------
        cfg.err.RUNTM_DRV
            If reference images are not set.

        Notes
        -----
        Also appends the relevant points to the `cohtraces`.
        """
        if not self._ref_are_set:
            raise ERR.RUNTM_DRV(ERR.RUNTM_DRV.str("reference images not set"))
        image_zero_mean = np.subtract(
            np_image, self._ref_mean_image, dtype="float64"
        )
        # custom divide: replace nan by zero
        normview = np.divide(
            image_zero_mean,
            self._ref_norm_image,
            out=np.zeros_like(image_zero_mean),
            where=(self._ref_norm_image != 0),
            dtype="float64"
        )
        self._normalized_images.append(normview)
        self.add_image_records(normview)
        self._cohtrace_access.acquire()
        for it in range(len(self._cohtrace_data)):
            self._cohtrace_data[it].append(normview[self._cohtrace_coords[it]])
        self._cohtrace_access.release()

    def add_image_records(self, normview):
        """
        Processes a normalized image for the (cropped) image records.

        Parameters
        ----------
        normview : numpy.ndarray(dtype=float64)
            Normalized image.
        """
        # Counters: i, j: indices; x, y: raw coordinates
        for i, x in enumerate(
            np.arange(self._crop_coords[0, 0], self._crop_coords[1, 0])
        ):
            for j, y in enumerate(
                np.arange(self._crop_coords[0, 1], self._crop_coords[1, 1])
            ):
                if normview[x, y] > self._image_records["max"][i, j]:
                    self._image_records["max"][i, j] = normview[x, y]
                    self._image_pos["max"][i, j] = (
                        self._piezo_trace[self._piezo_trace_counter]
                    )
                if normview[x, y] < self._image_records["min"][i, j]:
                    self._image_records["min"][i, j] = normview[x, y]
                    self._image_pos["min"][i, j] = (
                        self._piezo_trace[self._piezo_trace_counter]
                    )
        self._image_records["sum"] = normview[
            self._crop_coords[0, 0]:self._crop_coords[1, 0],
            self._crop_coords[0, 1]:self._crop_coords[1, 1]
        ]
        self._image_record_counter += 1

    def set_ref_image(self, np_image, ref_arm):
        """
        Sets the reference image attributes and calculates the normalization.

        Parameters
        ----------
        np_image : numpy.ndarray(dtype=uint8)
            8-bit grayscale image from camera.
        ref_arm : str
            "fixed", "scanned".

        Raises
        ------
        cfg.err.RUNTM_DRV
            If parameters are invalid.
        """
        if ref_arm == "fixed":
            self._ref_fixed_image = np_image
        elif ref_arm == "scanned":
            self._ref_scanned_image = np_image
        else:
            raise ERR.RUNTM_DRV(ERR.RUNTM_DRV.str())
        self._update_ref_images()


###############################################################################


if __name__ == "__main__":

    # Test settings
    step_time = 0.1
    cohtraces = 3
    piezo_steps = 100
    piezo_trace = "linear_up"
    piezo_port = "COM2"

    # Run app
    app = QApplication(sys.argv)
    camera_cfg = get_camera_cfg()
    piezo_cfg = get_piezo_cfg(port=piezo_port)
    coh_meas = CohMeas(
        camera_cfg, piezo_cfg,
        step_time=step_time, piezo_steps=piezo_steps,
        piezo_trace=piezo_trace, cohtraces=cohtraces
    )
    coh_meas.show()
    app_ret = app.exec_()

    # Clean up
    coh_meas.shutdown()
    sys.exit(app_ret)
