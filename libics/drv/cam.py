# System Imports
import threading
import time

# Package Imports
try:
    from . import addpath   # noqa
except(ImportError):
    import addpath          # noqa
import util.thread as thread

# Subpackage Imports
import drv.camutil as camutil
import drv.itf.camcfg as camcfg


###############################################################################


class Camera(object):

    """
    Wrapper for video features.

    Supports sequential and asynchronous capture. Features frame buffers and
    timed callbacks.

    Parameters
    ----------
    camera_cfg : itf.camcfg.CameraCfg
        Camera capture configuration.
    """

    def __init__(self, camera_cfg=camcfg.CameraCfg()):
        self._camera_origin = camutil.CameraOrigin(camera_cfg)
        self._matrix_data = None        # empty mdata storing only metadata
        # Frame buffer
        self._frame_buffer_enabled = False
        self._frame_buffer = []         # cache for retrieved images
        self._frame_buffer_lock = threading.Lock()
        # Callback functions
        self._origin_callback = None    # on video origin callback
        self._timer_callback = {}       # dict: id -> util.thread.PeriodicTimer

    # ++++ Origin callback ++++++++++++++++++++++

    def set_origin_callback(self, func):
        """
        Sets a callback function that is called when a frame is ready.

        Parameters
        ----------
        func : callable or None
            If callable, call signature: `func(numpy.ndarray)`.
            If `None`, removes origin callback.
        """
        if callable(func):
            self._origin_callback = func

    # ++++ Timer callback +++++++++++++++++++++++

    def add_timer(self, timer_id, timeout, func, *args, **kwargs):
        """
        Adds a timer that calls a function when it times out.

        Parameters
        ----------
        timer_id : str
            ID of timer with which the timer can be referenced.
        timeout : int
            Timeout in ms.
        func : callable
            Call signature: `func()`.
        *args, **kwargs
            Arguments that are passed to `func`.

        Notes
        -----
        Remember to acquire the lock if the worker function makes use of
        the frame buffer feature.
        """
        if timer_id in self._timer_callback.keys():
            self.remove_timer(timer_id=timer_id)
        self._timer_callback[timer_id] = thread.PeriodicTimer(
            timeout, func, *args, **kwargs
        )

    def remove_timer(self, timer_id=None):
        """
        Removes a timer with given ID. If `None`, removes all timers.
        """
        self.stop_timer(timer_id=timer_id)
        if timer_id is None:
            self._timer_callback = {}
        else:
            del self._timer_callback[timer_id]

    def start_timer(self, timer_id=None):
        """
        Starts a timer with given ID. If `None`, starts all timers.
        """
        if timer_id is None:
            for _, t in self._timer_callback.items():
                t.start()
        else:
            self._timer_callback[timer_id].start()

    def stop_timer(self, timer_id=None):
        """
        Stops a timer with given ID. If `None`, stops all timers.
        """
        if timer_id is None:
            for _, t in self._timer_callback.items():
                t.stop()
        else:
            self._timer_callback[timer_id].stop()

    # ++++ Frame buffer +++++++++++++++++++++++++

    def enable_frame_buffer(self):
        """
        Enables the frame buffer feature.

        Captured images are stored in a list, so a time history of images is
        visible. In order not to fill up memory, the frame buffer must be
        regularly cleared using the `reset_frame_buffer()` method.

        Notes
        -----
        Typical usage in combination with a timer: Every timeout period the
        time history of images within the period is analyzed, then the buffered
        frames are cleared.
        """
        self.acquire_lock()
        self._frame_buffer_enabled = True
        self.release_lock()

    def disable_frame_buffer(self):
        """
        Disables the frame buffer feature.
        """
        self.acquire_lock()
        self._frame_buffer_enabled = False
        self.release_lock()

    def reset_frame_buffer(self):
        """
        Clears the buffered frames to release memory. NOT THREAD SAFE!

        Notes
        -----
        Before running this function, call `acquire_lock()`.
        """
        self._frame_buffer = []

    def get_frame_buffer(self):
        """
        Gets the frame buffer. NOT THREAD SAFE!

        Returns
        -------
        frame_buffer : list(numpy.ndarray)
            The frame buffer.

        Notes
        -----
        Before running this function, call `acquire_lock()`.
        After processing of the frame buffer itself (i.e. e.g. copying
        the frames into local variables) has finished, call `release_lock()`.
        """
        return self._frame_buffer

    def acquire_lock(self):
        """
        Lock for frame buffer access.
        """
        self._frame_buffer_lock.acquire()

    def release_lock(self):
        """
        Lock for frame buffer access.
        """
        self._frame_buffer_lock.release()

    def _load_frame_buffer(self, frame, *args, **kwargs):
        """
        Appends a frame to the frame buffer.

        Parameters
        ----------
        frame : numpy.ndarray
            Frame as `numpy.ndarray` where the dimensions typically
            correspond to (width, height, channel).
        *args, **kwargs
            Currently ignored.
        """
        self.acquire_lock()
        self._frame_buffer.append(frame)
        self.release_lock()

    # ++++ Metadata +++++++++++++++++++++++++++++

    def set_matrix_data(self, mdata):
        self._matrix_data = mdata

    def get_matrix_data(self):
        return self._matrix_data

    def set_camera_cfg(self, camera_cfg):
        self._camera_origin.set_camera_cfg(camera_cfg)

    def get_camera_cfg(self):
        return self._camera_origin.get_camera_cfg()

    # ++++ Capture setup ++++++++++++++++++++++++

    def _cb_origin(self, frame, *args, **kwargs):
        """
        Callback function on frame ready.
        """
        if self._frame_buffer_enabled:
            self._load_frame_buffer(frame, *args, **kwargs)
        if self._origin_callback is not None:
            self._origin_callback(frame)

    def open_camera(self):
        """
        Sets up the camera interface.
        """
        return self._camera_origin.open_camera()

    def close_camera(self):
        """
        Closes the camera interface.
        """
        return self._camera_origin.close_camera()

    def read_camera_cfg(self, overwrite_cfg=False):
        """
        Loads the camera configuration from the camera.

        Parameters
        ----------
        overwrite_cfg : bool
            Whether to overwrite the configuration stored in
            `camera_origin`.

        Returns
        -------
        camera_cfg : drv.itf.camcfg.CameraCfg
            Currently active camera configuration
        """
        camera_cfg = self._camera_origin.read_camera_cfg(
            overwrite_cfg=overwrite_cfg
        )
        if overwrite_cfg:
            self._camera_origin.set_camera_cfg(camera_cfg)
        return camera_cfg

    def write_camera_cfg(self):
        """
        Writes the camera configuration stored in `camera_origin` to the
        actual camera.

        Notes
        -----
        Writes only flagged attributes.
        """
        self._camera_origin.write_camera_cfg()

    def run(self):
        """
        Starts capturing.
        """
        return self._camera_origin.run(callback=self._cb_origin)

    def stop(self):
        """
        Stops capturing.
        """
        return self._camera_origin.stop()

    def grab(self):
        """
        Grabs the latest frame.

        Returns
        -------
        frame : numpy.ndarray or None
            `None` if no frame available.
        """
        frame = None
        if self._frame_buffer_enabled:
            self.acquire_lock()
            if len(self._frame_buffer) > 0:
                frame = self._frame_buffer[-1]
            self.release_lock()
        else:
            frame = self._camera_origin.grab()
        return frame


###############################################################################


if __name__ == "__main__":

    # Create test config for Vimba Manta camera
    camera_cfg = camcfg.CameraCfg()
    camera_cfg.camera.camera_type.val = "vimba"

    # Create dummy callback functions
    def print_origin_cb(frame):
        print("Origin callback: cb image shape =", frame.shape)

    def print_timer_cb(camera):
        print("3s timer: grabbed image shape =", camera.grab().shape)
        camera.reset_frame_buffer()

    # Create camera object
    camera = Camera(camera_cfg=camera_cfg)
    camera.set_origin_callback(print_origin_cb)
    camera.add_timer("3sTimer", 3000, print_timer_cb, camera)
    camera.enable_frame_buffer()

    # Configure camera
    camera.open_camera()
    camera.read_camera_cfg(overwrite_cfg=True)

    # Capture images
    camera.run()
    camera.start_timer()
    time.sleep(20000)
    camera.stop_timer()
    camera.stop()

    # Cleanup
    camera.close_camera()
