import abc
import collections

import numpy as np

from libics.drv.itf import itf, vimba


###############################################################################


def get_bin_itf(cfg):
    if cfg.interface == itf.ITF_BIN.VIMBA:
        return VimbaItf(cfg)


class BinItfBase(abc.ABC):

    def __init__(self, cfg=None):
        self.cfg = cfg

    @abc.abstractmethod
    def setup(self):
        pass

    @abc.abstractmethod
    def shutdown(self):
        pass

    @abc.abstractmethod
    def connect(self):
        pass

    @abc.abstractmethod
    def close(self):
        pass


###############################################################################


class VimbaItf(BinItfBase):

    def __init__(self, cfg):
        super().__init__(cfg)
        self._camera = None
        self._frames = {}
        self._requeue = {}
        self._callback = {}
        self.__index_counter = -1
        self.__latest_index = -1
        self.__queue_order = collections.deque([-1], maxlen=1)
        self.__frame_buffer = {}

    def setup(self):
        vimba.startup()
        if self.cfg.device is None:
            cams = vimba.get_vimba_cameras()
            if len(cams) == 0:
                raise NameError("Vimba device invalid")
            elif len(cams) > 1:
                print("Warning: multiple Vimba cameras found")
            self._camera = cams[0]._camera
        else:
            self._camera = vimba.getVimba().getCamera(self.cfg.device)

    def shutdown(self):
        vimba.shutdown()

    def connect(self):
        self._camera.openCamera()

    def close(self):
        self._camera.closeCamera()

    @property
    def latest_index(self):
        """
        Gets the last updated index.
        """
        return self.__latest_index

    @property
    def next_index(self):
        """
        Gets the frame index next to be filled.
        """
        try:
            return self.__queue_order[0]
        except IndexError:
            return self.__index_counter

    @property
    def cam(self):
        """
        Convenience access to pymba camera object.
        """
        return self._camera

    def setup_frames(self, callback=None):
        """
        Adds and announces frames according to the configuration object.

        Parameters
        ----------
        callback : callable or None
            Function that is called when the frame is ready.
            Must take the raw image data (buffer protocol) as
            parameter.
            If `callable` is not callable, no callback function
            will be called.
        """
        for _ in range(self.cfg.frame_count):
            self.add_frame(requeue=self.cfg.frame_requeue, callback=callback)
        self.announce_frame()

    # ++++ Wrapper methods ++++++++++++++

    def add_frame(self, requeue=True, callback=None):
        """
        Creates a new Vimba API frame.

        Parameters
        ----------
        requeue : bool
            Whether to automatically requeue the frame after its data
            has been retrieved.
        callback : callable or None
            Function that is called when the frame is ready.
            Must take the raw image data (buffer protocol) as
            parameter.
            If `callable` is not callable, no callback function
            will be called.
        """
        self.__index_counter += 1
        self.__queue_order = collections.deque(maxlen=len(self._frames))
        self._frames[self.__index_counter] = self._camera.getFrame()
        self._requeue[self.__index_counter] = requeue
        self._callback[self.__index_counter] = lambda frame: (
            self._on_frame_ready(self.__index_counter, callback, frame)
        )

    def revoke_all_frames(self):
        """
        Revokes all frames from the Vimba API.
        """
        self._camera.revokeAllFrames()
        self._frames = {}
        self._requeue = {}
        self._callback = {}
        self.__queue_order = collections.deque([-1], maxlen=1)

    def start_capture(self):
        """
        Prepares the Vimba API for incoming frames.
        """
        self._camera.startCapture()
        for key, val in self._requeue.items():
            if val:
                self.queue_frame_capture(index=key)

    def end_capture(self):
        """
        Stops the Vimba API from being able to receive frames.
        """
        self._camera.endCapture()

    def start_acquisition(self):
        """
        Starts frame acquisition.
        """
        self._camera.runFeatureCommand("AcquisitionStart")

    def end_acquisition(self):
        """
        Ends frame acquisition.
        """
        self._camera.runFeatureCommand("AcquisitionStop")

    def flush_capture_queue(self):
        """
        Flushes the capture queue.
        """
        self._camera.flushCaptureQueue()

    def announce_frame(self, index=None):
        """
        Announces a frame to the Vimba API that may be queued for frame
        capturing later.

        Vimba C API: Runs VmbFrameAnnounce.
        """
        err = None
        if index is None:
            err = self._do_for_all_frames(self.announce_frame)
        else:
            err = self._frames[index].announceFrame()
        return err

    def revoke_frame(self, index=None):
        """
        Revokes a frame from the Vimba API.
        """
        err = None
        if index is None:
            err = self._do_for_all_frames(self.revoke_frame)
        else:
            err = self._frames[index].revokeFrame()
            del self._frames[index]
            del self._requeue[index]
            del self._callback[index]
            _queue = np.array(self.__queue_order)
            _queue = _queue[_queue != index]
            self.__queue_order = collections.deque(
                _queue, maxlen=len(self._frames)
            )
        return err

    def queue_frame_capture(self, index=None):
        """
        Queues a frame that may be filled during frame capturing.

        Vimba C API: Runs VmbCaptureFrameQueue.
        """
        err = None
        if index is None:
            err = self._do_for_all_frames(self.queue_frame_capture)
        else:
            err = self._frames[index].queueFrameCapture(
                frameCallback=self._callback[index]
            )
            self.__queue_order.append(index)
        return err

    def wait_frame_capture(self, index=None, timeout=2.0):
        """
        Waits for a queued frame to be filled (or dequeued).

        Vimba C API: Runs VmbCaptureFrameWait.

        Parameters
        ----------
        timeout : float
            Waiting timeout in seconds (s).
        """
        err = None
        if index is None:
            err = self._do_for_all_frames(
                self.wait_frame_capture, timeout=timeout
            )
        else:
            err = self._frames[index].waitFrameCapture(
                timeout=int(1000 * timeout)
            )
        return err

    def get_buffer_byte_data(self, index=None):
        """
        Calls the Vimba API to obtain the data of a frame.

        Returns
        -------
        data : PyObject (Py_buffer)
            Raw image data.
        """
        data = None
        if index is None:
            data = self._do_for_all_frames(self.get_buffer_byte_data)
        else:
            data = self._frames[index].getBufferByteData()
            self.__frame_buffer[index] = data
            self.__latest_index = index
            if self._requeue[index]:
                self.queue_frame_capture(index=index)
        return data

    def grab_data(self, index=None):
        """
        Gets the most recently loaded frame buffer data.

        Returns
        -------
        data : PyObject (Py_buffer)
            Raw image data.
        """
        if index is None:
            return self._do_for_all_frames(self.grab_data)
        else:
            return self.__frame_buffer[index]

    # ++++ Helper methods +++++++++++++++

    def _do_for_all_frames(self, func, *args, **kwargs):
        """
        Applies a given function to all frames.

        Parameters
        ----------
        func : callable
            Function to be called.
            Must contain keyword argument `index`.
        *args, **kwargs
            Parameters to be passed to `func`.
        """
        ret = {}
        for key in self._frames.keys():
            ret[key] = func(*args, index=key, **kwargs)
        return ret

    def _on_frame_ready(self, index, callback, frame):
        """
        Callback function when the Vimba API has a frame ready.

        Parameters
        ----------
        index : int
            Internal index of frame.
        callback : callable or None
            Callback function. Only called if callable.
        frame : pymba.vimbaframe.VimbaFrame
            Pymba-wrapped Vimba API frame.
        """
        data = self.get_buffer_byte_data(index=index)
        if callable(callback):
            callback(data)
