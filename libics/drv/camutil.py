# Package Imports
try:
    from . import addpath   # noqa
except(ImportError):
    import addpath          # noqa
import cfg.err as ERR

# Subpackage Imports
import drv.itf.camcfg as camcfg
import drv.itf.vimba as vimba


###############################################################################


class CameraOrigin(object):

    """
    Function call distribution wrapper.

    Depending on which camera is opened, different setup, acquisition, etc.
    functions are called to obtain the same behaviour despite different
    (hardware) interfaces.

    Parameters
    ----------
    camera_cfg : drv.itf.camcfg.CameraCfg
        Camera configuration container determining camera and
        capturing settings.

    Raises
    ------
    cfg.err.DTYPE_CUSTOM
        If parameters are invalid.
    cfg.err.RUNTM_DRV_CAM
        If camera runtime error occurs.
    """

    def __init__(self, camera_cfg=camcfg.CameraCfg()):
        """
        FIXME: check with relative imports
        ERR.assertion(
            ERR.DTYPE_CUSTOM,
            type(camera_cfg) == camcfg.CameraCfg,
            description="data type: expected drv.itf.camcfg.CameraCfg"
        )
        """
        self._camera_cfg = camera_cfg
        self._camera = None
        self._init_camera()
        self._grab_func = None

    def _init_camera(self):
        """
        Creates and initializes the respective camera object.
        """
        if self._camera_cfg.camera.camera_type.val == "vimba":
            self._camera = _init_camera_vimba(self._camera_cfg)
        else:
            raise ERR.RUNTM_DRV_CAM(ERR.RUNTM_DRV_CAM.str())

    # ++++ Camera interface ++++++++++++++++++++++++++++

    def get_camera(self):
        """
        Gets the underlying camera interface.
        """
        return self._camera

    def open_camera(self):
        """
        Opens camera interface.
        """
        print("camutil: open_camera")
        if self._camera_cfg.camera.camera_type.val == "vimba":
            _open_camera_vimba(self._camera)

    def close_camera(self):
        """
        Closes camera interface.
        """
        print("camutil: close_camera")
        if self._camera_cfg.camera.camera_type.val == "vimba":
            _close_camera_vimba(self._camera)

    # ++++ Camera configuration ++++++++++++++++++++++++

    def set_camera_cfg(self, camera_cfg):
        """
        Sets the `camera_cfg` attribute. Only sets the differing update flags.
        """
        print("camutil: set_camera_cfg")
        self._camera_cfg.set_config(camera_cfg)

    def get_camera_cfg(self):
        """
        Gets the `camera_cfg` attribute.
        """
        return self._camera_cfg

    def read_camera_cfg(self):
        """
        Reads and gets the actual camera configuration.
        Does NOT overwrite the `camera_cfg` attribute.
        """
        print("camutil: read_camera_cfg")
        camera_cfg = None
        if self._camera_cfg.camera.camera_type.val == "vimba":
            camera_cfg = _read_camera_cfg_vimba(self._camera)
        return camera_cfg

    def write_camera_cfg(self):
        """
        Writes the `camera_cfg` attributes' configuration into the camera. Only
        writes the properties with set update flags.
        """
        print("camutil: write_camera_cfg")
        if self._camera_cfg.camera.camera_type.val == "vimba":
            _write_camera_cfg_vimba(self._camera, self._camera_cfg)

    # ++++ Capturing +++++++++++++++++++++++++++++++++++

    def grab(self):
        """
        Gets the current image from the camera interface.
        """
        return self._grab_func()

    def run(self, callback=None):
        """
        Starts capturing. If not `None`, then `callback` function is called
        on every frame ready event. The call signature is
        `callback(numpy.ndarray)`.
        """
        if self._camera_cfg.camera.camera_type.val == "vimba":
            self._grab_func = _run_vimba(self._camera, callback=callback)

    def stop(self):
        """
        Stops capturing.
        """
        if self._camera_cfg.camera.camera_type.val == "vimba":
            _stop_vimba(self._camera)


###############################################################################
# Initialization
###############################################################################

# ++++++++++ Vimba +++++++++++++++++++++++++++++


def _init_camera_vimba(camera_cfg):
    vimba.startup()
    cameras = vimba.get_vimba_cameras(
        regex_id_filter=camera_cfg.camera.camera_id.val
    )
    if len(cameras) == 0:
        raise ERR.RUNTM_DRV_CAM(ERR.RUNTM_DRV_CAM.str())
    else:
        if len(cameras) > 1:
            print("libics.drv.camutil._init_camera:" +
                  "non-unique Vimba camera ID")
        cam = cameras[0]
        camera_cfg.camera.camera_id = cam.get_id()
        return cam


###############################################################################
# Setup
###############################################################################

# ++++++++++ Vimba +++++++++++++++++++++++++++++


def _open_camera_vimba(camera):
    camera.open_cam(mode="full")


def _close_camera_vimba(camera):
    camera.close_cam()


###############################################################################
# Configuration
###############################################################################

# ++++++++++ Vimba +++++++++++++++++++++++++++++


def _read_camera_cfg_vimba(camera):
    camera_cfg = camcfg.CameraCfg()

    mode, multi_count = camera.get_acquisition()
    if mode == "Continuous":
        camera_cfg.acquisition.frame_count.val = 0
    elif mode == "SingleFrame":
        camera_cfg.acquisition.frame_count.val = 1
    elif mode == "MultiFrame":
        camera_cfg.acquisition.frame_count.val = multi_count

    width, height, width_offset, height_offset, px_format = camera.get_format()
    camera_cfg.image_format.width.val = width
    camera_cfg.image_format.height.val = height
    camera_cfg.image_format.width_offset.val = width_offset
    camera_cfg.image_format.height_offset.val = height_offset
    if px_format == "Mono8":
        camera_cfg.image_format.channel.val = "mono"
        camera_cfg.image_format.bpc.val = 8
    elif px_format == "Mono12":
        camera_cfg.image_format.channel.val = "mono"
        camera_cfg.image_format.channel.val = 12

    auto, time, nir_mode = camera.get_exposure()
    if auto == "Off":
        camera_cfg.exposure.cam_auto.val = "off"
    elif auto == "Single":
        camera_cfg.exposure.cam_auto.val = "single"
    elif auto == "Continuous":
        camera_cfg.exposure.cam_auto.val = "on"
    camera_cfg.exposure.time.val = float(time) * 1e-6
    if nir_mode == "Off":
        camera_cfg.acquisition.color_sensitivity.val = "normal"
    elif nir_mode == "On_HighQuality":
        camera_cfg.acquisition.color_sensitivity.val = "nir_hq"
    elif nir_mode == "On_Fast":
        camera_cfg.acquisition.color_sensitivity.val = "nir_fast"

    return camera_cfg


def _write_camera_cfg_vimba(camera, camera_cfg):
    print("write_camera_cfg_vimba")
    mode, multi_count = None, None
    if camera_cfg.acquisition.frame_count.flag:
        if camera_cfg.acquisition.frame_count.val == 0:
            mode = "Continuous"
        elif camera_cfg.acquisition.frame_count.val == 1:
            mode = "SingleFrame"
        else:
            mode = "MultiFrame"
    if camera_cfg.acquisition.frame_count.flag:
        multi_count = camera_cfg.acquisition.frame_count.val
    camera.set_acquisition(mode=mode, multi_count=multi_count)

    width, height, width_offset, height_offset = None, None, None, None
    px_format = None
    if camera_cfg.image_format.width.flag:
        width = camera_cfg.image_format.width.val
    if camera_cfg.image_format.height.flag:
        height = camera_cfg.image_format.height.val
    if camera_cfg.image_format.width_offset.flag:
        width_offset = camera_cfg.image_format.width_offset.val
    if camera_cfg.image_format.height_offset.flag:
        height_offset = camera_cfg.image_format.height_offset.val
    if camera_cfg.image_format.bpc.flag:
        if camera_cfg.image_format.bpc.val == 8:
            px_format = "Mono8"
        elif camera_cfg.image_format.bpc.val == 12:
            px_format = "Mono12"
    camera.set_format(width=width, height=height, width_offset=width_offset,
                      height_offset=height_offset, px_format=px_format)

    auto, time, nir_mode = None, None, None
    if camera_cfg.exposure.cam_auto.flag:
        if camera_cfg.exposure.cam_auto.val == "on":
            auto = "Continuous"
        elif camera_cfg.exposure.cam_auto.val == "single":
            auto = "Single"
        elif camera_cfg.exposure.cam_auto.val == "off":
            auto = "Off"
    if camera_cfg.exposure.time.flag:
        time = int(round(camera_cfg.exposure.time.val * 1e6))
    if camera_cfg.acquisition.color_sensitivity.flag:
        if camera_cfg.acquisition.color_sensitivity.val == "normal":
            nir_mode = "Off"
        elif camera_cfg.acquisition.color_sensitivity.val == "nir_fast":
            nir_mode = "On_Fast"
        elif camera_cfg.acquisition.color_sensitivity.val == "nir_hq":
            nir_mode = "On_HighQuality"
    camera.set_exposure(auto=auto, time=time, nir_mode=nir_mode)


###############################################################################
# Capturing
###############################################################################

# ++++++++++ Vimba +++++++++++++++++++++++++++++


def _run_vimba(camera, callback=None):
    camera.set_frame_buffers(count=3)
    camera.start_capture(frame_callback=callback)
    camera.start_acquisition()
    return camera.get_image


def _stop_vimba(camera):
    camera.end_acquisition()
    camera.end_capture()
