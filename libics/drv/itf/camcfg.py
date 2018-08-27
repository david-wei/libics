# Package Imports
try:
    from . import addpath   # noqa
except(ImportError):
    import addpath          # noqa
from util.types import FlaggedType


class CameraCfg(object):

    """
    Camera capture configuration container.

    Properties
    ----------
    * camera:
        - camera_type : str
            "vimba"
        - camera_id : str or None
            str: <camera_id>
            None: automatically finds any camera of camera_type
    * image_format:
        - width : positive int
        - height : positive int
        - width_offset : non-negative int
        - height_offset : non-negative int
        - channel : str or None
            str: "rgb", "rgba", "bgr", "bgra", "mono"
            None: "mono" but array without channel dimension
        - bpc (bits per channel) : int
            8, 10, 12
    * acquisition:
        - frame_count : non-negative int
            0: infinite (continuous acquisition)
            >0: finite (single or multi-frame acquisition)
        - color_sensitivity : str
            "normal", "nir_fast", "nir_hq"
    * exposure:
        - cam_auto : str
            "on": continuous auto-exposure
            "single": single auto-exposure, then constant
            "off": manual exposure
        - time : positive float
            float: exposure time in seconds

    Notes
    -----
    All attributes are of `util.types.FlaggedType` in which the flag is
    interpreted as a to-be-updated flag.

    See Also
    --------
    display.qtimage.QtImage.set_image_format
    """

    class Camera:

        CAMERA_TYPE = ["vimba"]

        def __init__(self, camera_type="vimba", camera_id=None):
            self.camera_type = FlaggedType(camera_type,
                                           cond=CameraCfg.Camera.CAMERA_TYPE)
            self.camera_id = FlaggedType(camera_id)

    class ImageFormat:

        CHANNEL = ["rgb", "rgba", "bgr", "bgra", "mono", None]
        BPC = [8, 10, 12]

        def __init__(self):
            self.width = FlaggedType(640)
            self.height = FlaggedType(480)
            self.width_offset = FlaggedType(0)
            self.height_offset = FlaggedType(0)
            self.channel = FlaggedType("mono",
                                       cond=CameraCfg.ImageFormat.CHANNEL)
            self.bpc = FlaggedType(8, cond=CameraCfg.ImageFormat.BPC)

    class Acquisition:

        COLOR_SENSITIVITY = ["normal", "nir_fast", "nir_hq"]

        def __init__(self):
            self.frame_count = FlaggedType(0)
            self.color_sensitivity = FlaggedType(
                "normal", cond=CameraCfg.Acquisition.COLOR_SENSITIVITY
            )

    class Exposure:

        CAM_AUTO = ["on", "single", "off"]

        def __init__(self):
            self.cam_auto = FlaggedType("off",
                                        cond=CameraCfg.Exposure.CAM_AUTO)
            self.time = FlaggedType(1e-3)

    def __init__(self, camera_type="vimba", camera_id=None):
        self.camera = CameraCfg.Camera(camera_type=camera_type,
                                       camera_id=camera_id)
        self.image_format = CameraCfg.ImageFormat()
        self.acquisition = CameraCfg.Acquisition()
        self.exposure = CameraCfg.Exposure()

    def set_all_flags(self, flag):
        """
        Sets the flags of all attributes to the given boolean value.
        """
        for cat_key, category in self.__dict__.items():
            if cat_key != "camera":
                for _, item in category.__dict__.items():
                    item.flag = flag

    def invert_all_flags(self):
        """
        Inverts the flags of all attributes.
        """
        for cat_key, category in self.__dict__.items():
            if cat_key != "camera":
                for _, item in category.__dict__.items():
                    item.invert()

    def set_config(self, camera_cfg, flags=None):
        """
        Sets the configuration parameters.

        If an attribute of the passed `camera_cfg` is `None`, this value is
        not set.

        Parameters
        ----------
        camera_cfg : CameraCfg
            The camera configuration to be set.
        flags : None or bool
            `None`: Sets differential update flags.
            `True`: Sets all update flags.
            `False`: Sets no update flags.
        """
        diff_flag = (flags is None)
        for cat_key, cat_val in self.__dict__.items():
            if cat_key != "camera":
                for item_key, item_val in cat_val.__dict__.items():
                    if item_val is not None:
                        if (self.__dict__[cat_key].__dict__[item_key]
                                != item_val):
                            self.__dict__[cat_key].__dict__[item_key].assign(
                                item_val, diff_flag=diff_flag
                            )
                        else:
                            (self.__dict__[cat_key].__dict__[item_key]
                             .flag) = False
        if type(flags) == bool:
            self.set_all_flags(flags)
