# System Imports
import numpy as np
import pymba
import re
import six
import time


###############################################################################
# Initialization
###############################################################################


# Global variable for Vimba API object
_Vimba = None
_VimbaSystem = None
TIMEOUT_DISCOVERY_GIGE = 0.2


def startup():
    """
    Initializes the Vimba C API.

    The Vimba API requires a startup and a shutdown call. This function checks
    whether startup has already been called.

    Returns
    -------
    _Vimba : pymba.vimba.Vimba
        Vimba API object.
    """
    global _Vimba, _VimbaSystem
    if _Vimba is None:
        _Vimba = pymba.vimba.Vimba()
        _Vimba.startup()
        _VimbaSystem = _Vimba.getSystem()
    return _Vimba


def shutdown():
    """
    Closes the Vimba C API.

    If using the Vimba API, this function must be called upon system return.

    Returns
    -------
    success : bool
        `True`: Vimba API was successfully closed.
        `False`: Vimba API was closed already.
    """
    global _Vimba, _VimbaSystem
    success = False
    if _Vimba is not None:
        _Vimba.shutdown()
        _Vimba = None
        _VimbaSystem = None
        success = True
    return success


def getVimba():
    """
    Gets the Vimba API object.

    Returns
    -------
    _Vimba : pymba.vimba.Vimba
        Vimba API object.
    """
    global _Vimba
    return _Vimba


###############################################################################
# Cameras
###############################################################################


class Camera(object):

    """
    Pure pythonic wrapper class for `pymba.vimbaCamera.VimbaCamera` with
    reduced functionality.

    Parameters
    ----------
    vimba_camera : pymba.vimbaCamera.VimbaCamera
        The actual Vimba camera object.

    Raises
    ------
    TypeError
        If parameters are invalid.
    """

    def __init__(self, vimba_camera):
        if type(vimba_camera) != pymba.vimbacamera.VimbaCamera:
            raise TypeError("invalid vimba_camera object")
        self._camera = vimba_camera
        self._is_open = False
        self._is_capturing = False
        self._is_acquiring = False
        self._frame_buffer = []
        self._frame_buffer_counter = 0
        self._px_dtype_size = 0
        self._px_rgb_size = 0

    def get_id(self):
        """
        Gets the camera ID string.

        Returns
        -------
        cam_id : str
            Camera ID string.
        """
        return self._camera.cameraIdString

    def get_info(self):
        """
        Gets the camera information.

        Returns
        -------
        info : tuple(str, str, str, str, int, str)
            (`cameraIdString`, `cameraName`, `modelName`,
            `serialString`, `permittedAccess`, `interfaceIdString`)
        """
        info = self._camera.getInfo()
        if six.PY3:
            info = (
                info.cameraIdString.decode("ascii"),
                info.cameraName.decode("ascii"),
                info.modelName.decode("ascii"),
                info.serialString.decode("ascii"),
                info.permittedAccess,
                info.interfaceIdString.decode("ascii")
            )
        else:
            info = (
                info.cameraIdString,
                info.cameraName,
                info.modelName,
                info.serialString,
                info.permittedAccess,
                info.interfaceIdString
            )
        return info

    @classmethod
    def _access_mode(cls, mode_str):
        if mode_str is None or mode_str == "none":
            return 0
        elif mode_str == "full":
            return 1
        elif mode_str == "read":
            return 2
        elif mode_str == "config":
            return 3
        elif mode_str == "lite":
            return 4
        else:
            raise TypeError("invalid camera access mode")

    def _set_px_format(self, px_format):
        """
        Sets the internal pixel format variables needed for numpy array
        representation of images.

        Parameters
        ----------
        px_format : str
            "Mono8", "Mono12", "Mono12Packed"

        Returns
        -------
        px_dtype_size : int
            Bit size of subpixel data type (e.g. 8 for `uint8`).
        px_rgb_size : int
            Number of subpixels (e.g. 1 for monochromatic).
        """
        if px_format == "Mono8":
            self._px_dtype_size = 8
            self._px_rgb_size = 1
        elif px_format == "Mono12":
            self._px_dtype_size = 16
            self._px_rgb_size = 1
        elif px_format == "Mono12Packed":
            # FIXME: handling packed 12bit data
            pass
        return self._px_dtype_size, self._px_rgb_size

    def open_cam(self, mode="full"):
        """
        Initializes the camera, `close` the camera after use.

        Parameters
        ----------
        mode : str
            "none", "full", "read", "config", "lite".

        Raises
        ------
        TypeError
            If parameters are invalid.
        """
        self._camera.openCamera(
            cameraAccessMode=Camera._access_mode(mode)
        )
        self._is_open = True

    def close_cam(self):
        """
        Closes the camera.
        """
        self._camera.closeCamera()
        self._is_open = False

    # ++++ Configuration +++++++++++++++++++++

    def set_acquisition(self, mode="Continuous"):
        """
        Sets the acquisition mode, i.e. how many frames are recorded.

        Parameters
        ----------
        mode : str, optional
            "Continuous", "SingleFrame", "MultiFrame".
        multi_count : int
            Number of frames to be acquired in multiframe mode.
            Has no effect in the other modes.

        Raises
        ------
        TypeError
            If parameters are invalid.
        """
        if (mode == "Continuous" or mode == "SingleFrame"
                or mode == "MultiFrame"):
            self._camera.AcquisitionMode = mode
        else:
            raise TypeError("invalid frame acquisition mode")

    def get_acquisition(self):
        """
        Gets the acquisition mode (str).
        """
        return self._camera.AcquisitionMode

    def get_max_size(self):
        """
        Gets the maximum image size.

        Returns
        -------
        px_max_x, px_max_y : int
            Maximal pixel count in x and y direction.
        """
        return self._camera.WidthMax, self._camera.HeightMax

    def set_format(self, width=None, height=None,
                   width_offset=None, height_offset=None,
                   px_format=None):
        """
        Sets the current image pixel size.

        Parameters
        ----------
        width, height, width_offset, height_offset : int or None
            Pixel counts. `None` does not set the variable.
            `width` and `height` may be `"max"` with which the
            camera's maximum image size is used and offsets are
            set to zero.
        px_format : str or None
            "Mono8", "Mono12", "Mono12Packed"
        """
        max_x, max_y = self.get_max_size()
        if width == "max":
            width = max_x
            width_offset = 0
        if height == "max":
            height = max_y
            height_offset = 0
        if width is not None and width <= max_x:
            self._camera.Width = width
        if height is not None and height < max_y:
            self._camera.Height = height
        width, height, _, __, ___ = self.get_format()
        if width_offset is not None:
            width_offset = max(min(max_x - width, width_offset), 0)
            self._camera.OffsetX = width_offset
        if height is not None:
            height_offset = max(min(max_y - height, height_offset), 0)
            self._camera.OffsetY = height_offset
        if (px_format is not None and
                (px_format == "Mono8" or px_format == "Mono12"
                 or px_format == "Mono12Packed")):
            self._camera.PixelFormat = px_format

    def get_format(self):
        """
        Gets the current image format.

        Returns
        -------
        width, height, width_offset, height_offset : int
            Pixel counts.
        px_format : str
            "Mono8", "Mono12", "Mono12Packed"
        """
        return (
            self._camera.Width, self._camera.Height,
            self._camera.OffsetX, self._camera.OffsetY,
            self._camera.PixelFormat
        )

    def set_exposure(self, auto=None, time=None, nir_mode=None):
        """
        Sets auto-exposure, manual exposure time and NIR exposure mode.

        The `None` parameter does not set the value.

        Parameters
        ----------
        auto : str or None
            "Off", "Single", "Continuous"
        time : int or None
            Exposure time in µs.
        nir_mode : str or None
            "Off", "On_HighQuality", "On_Fast"
        """
        if auto is not None and (auto == "Off" or auto == "Single"
                                 or auto == "Continuous"):
            self._camera.ExposureAuto = auto
        if time is not None and time > 0:
            self._camera.ExposureTimeAbs = time
        if nir_mode is not None and (nir_mode == "Off" or nir_mode == "On_Fast"
                                     or nir_mode == "On_HighQuality"):
            self._camera.NirMode = nir_mode

    def get_exposure(self):
        """
        Gets auto-exposure settings and manual exposure time in µs.
        """
        return self._camera.ExposureAuto, self._camera.ExposureTimeAbs

    # ++++ Image Capturing +++++++++++++++++++

    def set_frame_buffers(self, count=1):
        """
        Sets up frame buffers, into which the Vimba API may load received
        images. Automatically announces and revokes frames.

        Parameters
        ----------
        count : positive int
            Number of buffered frames.

        Raises
        ------
        ValueError
            If `count` value is invalid.
        """
        if count < 1:
            raise ValueError("invalid frame buffer count")
        rel_buffer_size = count - len(self._frame_buffer)
        # Drop frame buffers
        if rel_buffer_size < 0:
            for _ in range(-rel_buffer_size):
                self._frame_buffer[-1].revokeFrame()
                self._frame_buffer.pop()
        # Add frame buffers
        elif rel_buffer_size > 0:
            for _ in range(rel_buffer_size):
                self._frame_buffer.append(pymba.vimbaframe.VimbaFrame())
                self._frame_buffer[-1].announceFrame()

    def start_capture(self):
        self._camera.startCapture()
        self._frame_buffer_counter = 0
        for frame in self._frame_buffer:
            frame.queueFrameCapture()
        self._set_px_format(self._camera.PixelFormat)
        self._is_capturing = True

    def end_capture(self):
        self._camera.endCapture()
        self._camera.revokeAllFrames()
        self._is_capturing = False

    def start_acquisition(self):
        self._camera.runFeatureCommand("AcquisitionStart")
        self._is_acquiring = True

    def end_acquisition(self):
        self._camera.runFeatureCommand("AcquisitionStop")
        self._is_acquiring = False

    def get_image(self, index=None):
        """
        Gets an image stored in the frame buffer.

        Parameters
        ----------
        index : int or None
            Index of frame buffer list. `None` uses the frame which
            is subsequent to the internal frame buffer index.

        Returns
        -------
        image : numpy.ndarray or None
            Image with shape (height, width, rgb_size) and data type
            as defined by `PixelFormat`.
            `None` if no frame was captured.

        Notes
        -----
        Due to the internal frame buffer counter, subsequent images can be
        obtained by repeatedly calling `get_image()`.
        """
        # Choose frame buffer to be read
        if index is None:
            index = self._frame_buffer_counter + 1
        index = index % len(self._frame_buffer)
        frame = self._frame_buffer[index]
        # Wait for Vimba API response
        if frame.waitFrameCapture() == 0:
            # Load into numpy array
            image = np.array(
                buffer=frame.getBufferByteData(),
                dtype=np.uint8,
                shape=(frame.height, frame.width, 1)
            )
            self._frame_buffer_counter = index
            return image
        else:
            return None


def get_cameras(regex_id_filter=None):
    """
    Gets all discovered Vimba cameras.

    Parameters
    ----------
    regex_id_filter : str (regex) or None, optional
        `str`: Gets camera only if its ID matches the given regular expression.
        `None`: Gets all discovered cameras.

    Returns
    -------
    cameras : list(Camera)
        Discovered (and filtered) cameras. Note that these
        cameras are unopened.
    """
    global _Vimba, _VimbaSystem
    # Issue command to discover all GigE cameras
    if _VimbaSystem.GeVTLIsPresent:
        _VimbaSystem.runFeatureCommand("GeVDiscoveryAllOnce")
        time.sleep(TIMEOUT_DISCOVERY_GIGE)
    # Get all cameras
    camera_ids = _Vimba.getCameraIds()
    cameras = []
    if regex_id_filter is None:
        cameras = [Camera(_Vimba.getCamera(cam_id))
                   for cam_id in camera_ids]
    else:
        for cam_id in camera_ids:
            if re.match(regex_id_filter) is not None:
                cameras.append(Camera(_Vimba.getCamera(cam_id)))
    return cameras
