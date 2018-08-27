# System Imports
import collections
import numpy as np
import sys

# Qt Imports
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import (
    QApplication, QPushButton, QHBoxLayout, QVBoxLayout, QWidget
)

# Package Imports
import addpath          # noqa
from libics.display import qtimage
from libics.drv import cam
from libics.drv.itf import camcfg


###############################################################################


class ImageProcessor(QWidget, object):

    def __init__(self, camera_cfg, buffer_time=0.5, coh_hist_len=1000):
        super().__init__()
        self._buffer_time = buffer_time
        self._init_camera_cfg(camera_cfg)
        self._init_camera()
        self._init_qt_image()
        self._init_coh_hist(coh_hist_len)
        self._init_ui()
        self._init_connections()

    def _init_camera_cfg(self, camera_cfg):
        self.camera_cfg = camera_cfg

    def _init_camera(self):
        self.camera = cam.Camera(camera_cfg=camcfg.CameraCfg(
            camera_type=self.camera_cfg.camera.camera_type.val,
            camera_id=self.camera_cfg.camera.camera_id.val
        ))
        self.camera.set_origin_callback(self.display_frame)
        self.camera.add_timer(
            "proc_coh", int(1000 * self._buffer_time),
            self.process_frame_buffer
        )
        self.camera.enable_frame_buffer()

    def _init_qt_image(self):
        self.qt_image = qtimage.QtImage()
        self.qt_image.set_image_format(
            channel=self.camera_cfg.image_format.channel.val,
            bpc=self.camera_cfg.image_format.bpc.val
        )

    def _init_coh_hist(self, coh_hist_len):
        self._coh_hist_std = collections.deque(maxlen=coh_hist_len)
        self.coh_hist_std_image = qtimage.QtImage()
        self.coh_hist_std_image.set_image_format(channel=None, bpc=8)
        self._coh_hist_max = collections.deque(maxlen=coh_hist_len)
        self.coh_hist_max_image = qtimage.QtImage()
        self.coh_hist_max_image.set_image_format(channel=None, bpc=8)
        self._coh_hist_min = collections.deque(maxlen=coh_hist_len)
        self.coh_hist_min_image = qtimage.QtImage()
        self.coh_hist_min_image.set_image_format(channel=None, bpc=8)

    # ++++ GUI ++++++++++++++++++++++++++++++++++++

    def _init_ui(self):
        self.setWindowTitle("Reversing Wavefront Interferometer")
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        self._run_button = QPushButton("Run")
        self._stop_button = QPushButton("Stop")
        self._controls_layout = QHBoxLayout()
        self._controls_layout.addWidget(self._run_button)
        self._controls_layout.addWidget(self._stop_button)

        self.main_layout.addLayout(self._controls_layout)
        self.main_layout.addWidget(self.qt_image)
        self.main_layout.addWidget(self.coh_hist_std_image)
        self.main_layout.addWidget(self.coh_hist_max_image)
        self.main_layout.addWidget(self.coh_hist_min_image)

    def _init_connections(self):
        self._run_button.clicked.connect(self.run)
        self._stop_button.clicked.connect(self.stop)

    def show_gui(self):
        self.qt_image.show()
        self.coh_hist_std_image.show()
        self.coh_hist_max_image.show()
        self.coh_hist_min_image.show()

    # ++++ Setup capture ++++++++++++++++++++++++++

    def setup_camera(self):
        self.camera.open_camera()
        # Get current camera config
        self.camera.read_camera_cfg(overwrite_cfg=True)
        # Set target camera config
        self.camera.set_camera_cfg(self.camera_cfg)
        self.camera.write_camera_cfg()
        self.camera.close_camera()

    def open_camera(self):
        self.camera.open_camera()

    def close_camera(self):
        self.camera.close_camera()

    @pyqtSlot()
    def run(self):
        self.camera.run()

    @pyqtSlot()
    def stop(self):
        self.camera.stop()

    # ++++ Callback functions +++++++++++++++++++++

    def display_frame(self, np_image):
        """
        Shows a live preview of the camera image.
        """
        self.qt_image.update_image(np_image)

    def display_coh_hist_image(self):
        """
        Updates the coherence history images.
        """
        self.coh_hist_std_image.update_image(
            np.array(self._coh_hist_std, dtype="uint8")
        )
        self.coh_hist_max_image.update_image(
            np.array(self._coh_hist_max, dtype="uint8")
        )
        self.coh_hist_min_image.update_image(
            np.array(self._coh_hist_min, dtype="uint8")
        )

    def process_frame_buffer(self):
        """
        Extracts coherence properties of the last frame buffer.
        """
        self.camera.acquire_lock()
        im_buffer = np.array(self.camera.get_frame_buffer())
        print("Image shape:", im_buffer.shape)
        axes = None
        ch = self.camera_cfg.image_format.channel.val
        if (ch == "rgb" or ch == "rgba" or ch == "bgr" or ch == "bgra"
                or ch == "mono"):
            axes = (0, 1, 3)
        elif ch is None:
            axes = (0, 1)
        im_mean = np.mean(im_buffer, axis=axes)
        im_std = np.std(im_buffer, axis=axes)
        im_max = np.max(im_buffer, axis=axes)
        im_min = np.max(im_buffer, axis=axes)
        self._coh_hist_std.append(im_std / im_mean * 128 - 1)
        self._coh_hist_max.append(im_max / im_mean * 128 - 1)
        self._coh_hist_min.append(im_min / im_mean * 128 - 1)
        self.camera.reset_frame_buffer()
        self.camera.release_lock()


###############################################################################


def get_camera_cfg(
    camera_type="vimba", camera_id=None,
    line=None, color_sensitivity="normal",
    exposure_time=None
):
    camera_cfg = camcfg.CameraCfg()

    camera_cfg.camera.camera_type.set_val(camera_type, diff_flag=False)
    camera_cfg.camera.camera_id.set_val(camera_id, diff_flag=False)

    camera_cfg.image_format.width = None
    camera_cfg.image_format.height.set_val(1)
    camera_cfg.image_format.width_offset = None
    if line is None:
        camera_cfg.image_format.height_offset = None
    else:
        camera_cfg.image_format.height_offset.set_val(line)
    camera_cfg.image_format.channel.set_val("mono")
    camera_cfg.image_format.bpc.set_val(8)

    camera_cfg.acquisition.frame_count.set_val(0)
    camera_cfg.acquisition.color_sensitivity.set_val(color_sensitivity)

    camera_cfg.exposure.cam_auto.set_val("off")
    if exposure_time is None:
        camera_cfg.exposure.time = None
    else:
        camera_cfg.exposure.time.set_val(exposure_time)

    return camera_cfg


###############################################################################


if __name__ == "__main__":

    # Test settings
    line = 500
    buffer_time = 0.5
    coh_hist_len = 1000

    # Create Vimba ImageProcessor
    app = QApplication(sys.argv)
    camera_cfg = get_camera_cfg(line=line)
    im_proc = ImageProcessor(
        camera_cfg, buffer_time=buffer_time, coh_hist_len=coh_hist_len
    )
    im_proc.setup_camera()

    # Setup GUI
    im_proc.show()
    im_proc.show_gui()

    # Start previewer
    im_proc.open_camera()
    app_ret = app.exec_()
    im_proc.close_camera()
    sys.exit(app_ret)
