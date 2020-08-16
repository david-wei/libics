import abc
import numpy as np
import queue
import time
import threading

from libics.core.util.func import StoppableThread
from libics.driver.device import DevBase


###############################################################################


class FORMAT_COLOR:

    BW = "BW"
    GS = "GS"
    RGB = "RGB"
    RGBA = "RGBA"


class EXPOSURE_MODE:

    MANUAL = "MANUAL"
    CONTINUOS = "CONTINUOS"
    SINGLE = "SINGLE"


class SENSITIVITY:

    NORMAL = "NORMAL"
    NIR_FAST = "NIR_FAST"
    NIR_HQ = "NIR_HQ"


###############################################################################


class Camera(DevBase):

    """
    Attributes
    ----------

    **Configurations**

    frame_queue_size : `int`
        Maximum size of fast frame queue.

    **Properties**

    pixel_hrzt_count, pixel_vert_count : `int`
        Pixel count in respective direction.
    pixel_hrzt_size, pixel_vert_size : `float`
        Pixel size in meters in respective direction.
    pixel_hrzt_offset, pixel_vert_offset : `int`
        Offset of pixels to be captured.
    format_color : `FORMAT_COLOR`
        BW: black/white boolean image.
        GS: greyscale image.
        RGB: RGB color image.
        RGBA: RGB image with alpha channel.
    channel_bitdepth : `int`
        Bits per color channel.
    exposure_mode : `EXPOSURE_MODE`
        MANUAL: manual, fixed exposure time.
        CONTINUOS: continuosly self-adjusted exposure time.
        SINGLE: fixed exposure time after single adjustment.
    exposure_time : `float`
        Exposure time in seconds (s).
    acquisition_frames : `int`
        Number of frames to be acquired.
        0 (zero) is interpreted as infinite, i.e.
        continuos acquisition.
    """

    def __init__(self):
        super().__init__()
        self.frame_queue_size = 2
        self.properties.set_properties(**self._get_default_properties_dict(
            "pixel_hrzt_count", "pixel_vert_count",
            "pixel_hrzt_size", "pixel_vert_size",
            "pixel_hrzt_offset", "pixel_vert_offset",
            "format_color", "channel_bitdepth",
            "exposure_mode", "exposure_time", "acquisition_frames"
        ))
        self._is_running = threading.Lock()
        self._frame_queue = queue.Queue(maxsize=2)  # Fast buffer (thread-safe)
        self._frame_grab_thread = None              # Fast (handling I/O)
        self._frame_transfer_thread = None          # Slow (processing frame)
        self._last_frame = None                     # Slow buffer
        self._last_frame_lock = threading.Lock()    # Lock for slow buffer

    def configure(self, **cfg):
        super().configure(**cfg)
        if "frame_queue_size" in cfg:
            self._frame_queue = queue.Queue(maxsize=cfg["frame_queue_size"])

    # ++++++++++++++++++++++++++++++++++++++++
    # Camera methods
    # ++++++++++++++++++++++++++++++++++++++++

    def run(self, callback=None, blocking=False):
        """
        Start capturing images.

        Parameters
        ----------
        callback : `callable`
            Callback function to be called upon new frame ready.
        blocking : `bool`
            Flag whether to start acquisition in the calling thread.
        """
        if self.is_running():
            return
        self.free()
        # Slow thread
        if self._frame_transfer_thread is not None:
            self._frame_transfer_thread.stop()
            self._frame_transfer_thread = None
        self._frame_transfer_thread = StoppableThread(
            target=self._run_frame_transfer, kwargs={"callback": callback}
        )
        self._frame_transfer_thread.start()
        # Fast thread
        if self._frame_grab_thread is not None:
            self._frame_grab_thread.stop()
            self._frame_grab_thread = None
        if blocking:
            self._run_frame_grab()
        else:
            self._frame_grab_thread = StoppableThread(
                target=self._run_frame_grab
            )
            self._frame_grab_thread.start()

    def stop(self):
        """
        Stop capturing images.

        Only works if :py:meth:`run` was called non-blocking.
        """
        if self.is_running():
            if self._frame_grab_thread is not None:
                self._frame_grab_thread.stop()
            if self._frame_transfer_thread is not None:
                self._frame_transfer_thread.stop()
        self._frame_grab_thread = None
        self._frame_transfer_thread = None

    @abc.abstractmethod
    def _start_acquisition(self):
        """
        Issues the device to start acquisition.
        """

    @abc.abstractmethod
    def _end_acquisition(self):
        """
        Issues the device to end acquisition.
        """

    def is_running(self):
        """
        Checks whether camera is capturing.
        """
        return self._is_running.locked()

    @abc.abstractmethod
    def next(self):
        """
        Captures the next image and stores it in the frame queue.

        This method should be blocking.
        """

    def free(self):
        """
        Empties the frame queue.
        """
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                return

    def grab(self):
        """
        Grab image from camera (wait for next image).

        Returns
        -------
        image : `numpy.ndarray`
            Image as numpy array.
            Shape: height x width (x channel)
        """
        is_running = self.is_running()
        if not is_running:
            acquisition_frames = self.p.read("acquisition_frames")
            self.p.write(acquisition_frames=1)
            self.run()
        else:
            self.free()
        self.next()
        frame = self.get()
        if not is_running:
            self.stop()
            self.p.apply(acquisition_frames=acquisition_frames)
        return frame

    def get(self):
        """
        Get image from slow buffer (last taken image).

        Returns
        -------
        image : `numpy.ndarray`
            Image as numpy array.
            Shape: height x width (x channel)
        """
        self._last_frame_lock.acquire()
        frame = np.copy(self._last_frame)
        self._last_frame_lock.release()
        return frame

    def find_exposure_time(self, max_cutoff=2/3, max_iterations=64):
        """
        Varies the camera's exposure time to maximize image brightness
        while keeping the maximum ADC value below saturation.

        Parameters
        ----------
        max_cutoff : `float`
            Minimum relative brightness required for the image
            maximum.
        max_iterations : `int`
            Maximum number of iterations.

        Returns
        -------
        prev_exposure_time : `float`
            Previous exposure time in seconds (s).
            Allows the calling function to reset the settings.
        """
        prev_exposure_time = self.p.exposure_time
        max_adc = float(2**self.p.channel_bitdepth - 1)
        loop, verify = True, False
        it = -1
        while loop:
            it += 1
            im = self.grab()
            while im is None:
                im = self.grab()
            saturation = im.max() / max_adc
            if saturation >= 1:
                self.write_exposure_time(
                    val=self.p.exposure_time / 3
                )
                time.sleep(0.15)
                verify = False
            elif saturation < max_cutoff:
                scale = 1 / max_cutoff
                if saturation < 0.05:
                    scale = 3
                elif saturation < 0.85:
                    scale = max_cutoff / saturation
                self.p.write_exposure_time(
                    val=self.p.exposure_time * scale
                )
                time.sleep(0.15)
                verify = False
            else:
                if verify:
                    loop = False
                verify = True
            if it >= max_iterations:
                self.LOGGER.warning(
                    "finding exposure time: maximum iterations reached"
                )
                return prev_exposure_time
            self.LOGGER.debug(
                "Finding exposure time: {:.0f} µs (sat: {:.3f})"
                .format(self.p.exposure_time * 1e6, saturation)
            )
        return prev_exposure_time

    # ++++++++++++++++++++++++++++++++++++++++
    # Helper methods
    # ++++++++++++++++++++++++++++++++++++++++

    @property
    def _numpy_dtype(self):
        if self.p.channel_bitdepth == 1:
            return "bool"
        elif self.p.channel_bitdepth <= 8:
            return "uint8"
        elif self.p.channel_bitdepth <= 16:
            return "uint16"
        elif self.p.channel_bitdepth <= 32:
            return "uint32"

    @property
    def _numpy_shape(self):
        if self.p.format_color in (FORMAT_COLOR.BW, FORMAT_COLOR.GS):
            return (self.p.pixel_vert_count, self.p.pixel_hrzt_count)
        elif self.p.format_color == FORMAT_COLOR.RGB:
            return (self.p.pixel_vert_count, self.p.pixel_hrzt_count, 3)
        elif self.p.format_color == FORMAT_COLOR.RGBA:
            return (self.p.pixel_vert_count, self.p.pixel_hrzt_count, 4)

    def _cv_buffer_to_numpy(self, buffer):
        return np.ndarray(
            buffer=buffer,
            dtype=self._numpy_dtype,
            shape=self._numpy_shape
        )

    def _run_frame_grab(self):
        IN_THREAD = (
            self._frame_grab_thread is not None
            and self._frame_grab_thread.is_alive()
        )
        TIMEOUT_STOP = self.p.exposure_time / 3   # seconds
        self._start_acquisition()
        # Continuos acquisition
        if self.p.acquisition_frames == 0:
            while True:
                self.next()
                if IN_THREAD:
                    if self._frame_grab_thread.stop_event.wait(
                        timeout=TIMEOUT_STOP
                    ):
                        break
        # Manual acquisition
        else:
            for frame_no in range(self.p.acquisition_frames):
                self.next()
                if IN_THREAD:
                    if self._frame_grab_thread.stop_event.wait(
                        timeout=TIMEOUT_STOP
                    ):
                        break
        self._end_acquisition()

    def _run_frame_transfer(self, callback=None):
        TIMEOUT_FRAME = 1e-1   # seconds
        TIMEOUT_STOP = 1e-3    # seconds
        self._is_running.acquire()
        while True:
            try:
                frame = self._frame_queue.get(
                    block=True, timeout=TIMEOUT_FRAME
                )
                self._last_frame_lock.acquire()
                self._last_frame = np.copy(frame)
                self._last_frame_lock.release()
                if callback is not None:
                    callback(frame)
            except queue.Empty:
                pass
            if self._frame_transfer_thread.stop_event.wait(
                timeout=TIMEOUT_STOP
            ):
                break
        self._is_running.release()

    # ++++++++++++++++++++++++++++++++++++++++
    # Properties methods
    # ++++++++++++++++++++++++++++++++++++++++

    @abc.abstractmethod
    def read_pixel_hrzt_count(self):
        pass

    @abc.abstractmethod
    def write_pixel_hrzt_count(self, value):
        pass

    @abc.abstractmethod
    def read_pixel_hrzt_size(self):
        pass

    @abc.abstractmethod
    def write_pixel_hrzt_size(self, value):
        pass

    @abc.abstractmethod
    def read_pixel_hrzt_offset(self):
        pass

    @abc.abstractmethod
    def write_pixel_hrzt_offset(self, value):
        pass

    @abc.abstractmethod
    def read_pixel_vert_count(self):
        pass

    @abc.abstractmethod
    def write_pixel_vert_count(self, value):
        pass

    @abc.abstractmethod
    def read_pixel_vert_size(self):
        pass

    @abc.abstractmethod
    def write_pixel_vert_size(self, value):
        pass

    @abc.abstractmethod
    def read_pixel_vert_offset(self):
        pass

    @abc.abstractmethod
    def write_pixel_vert_offset(self, value):
        pass

    @abc.abstractmethod
    def read_format_color(self):
        pass

    @abc.abstractmethod
    def write_format_color(self, value):
        pass

    @abc.abstractmethod
    def read_channel_bitdepth(self):
        pass

    @abc.abstractmethod
    def write_channel_bitdepth(self, value):
        pass

    @abc.abstractmethod
    def read_exposure_mode(self):
        pass

    @abc.abstractmethod
    def write_exposure_mode(self, value):
        pass

    @abc.abstractmethod
    def read_exposure_time(self):
        pass

    @abc.abstractmethod
    def write_exposure_time(self, value):
        pass

    @abc.abstractmethod
    def read_acquisition_frames(self):
        pass

    @abc.abstractmethod
    def write_acquisition_frames(self, value):
        pass
