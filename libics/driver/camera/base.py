class DRV_CAM:

    class FORMAT_COLOR:

        BW = 0
        GS = 1
        RGB = 2
        RGBA = 3

    class EXPOSURE_MODE:

        MANUAL = 0
        CONTINUOS = 1
        SINGLE = 2

    class SENSITIVITY:

        NORMAL = 0
        NIR_FAST = 1
        NIR_HQ = 2


class CamCfg():

    """
    DrvCfgBase -> CamCfg.

    Parameters
    ----------
    pixel_hrzt_count, pixel_vert_count : int
        Pixel count in respective direction.
    pixel_hrzt_size, pixel_vert_size : float
        Pixel size in meters in respective direction.
    pixel_hrzt_offset, pixel_vert_offset : int
        Offset of pixels to be captured.
    format_color : DRV_CAM.FORMAT_COLOR
        BW: black/white boolean image.
        GS: greyscale image.
        RGB: RGB color image.
        RGBA: RGB image with alpha channel.
    channel_bitdepth : int
        Bits per color channel.
    exposure_mode : DRV_CAM.EXPOSURE_MODE
        MANUAL: manual, fixed exposure time.
        CONTINUOS: continuosly self-adjusted exposure time.
        SINGLE: fixed exposure time after single adjustment.
    exposure_time : float
        Exposure time in seconds (s).
    acquisition_frames : int
        Number of frames to be acquired.
        0 (zero) is interpreted as infinite, i.e.
        continuos acquisition.
    sensitivity : DRV_CAM.SENSITIVITY
        NORMAL: normal acquisition.
        NIR_FAST: fast near-IR enhancement.
        NIR_HQ: high-quality near-IR enhancement.

    Notes
    -----
    * Horizontal (hrzt) and vertical (vert) directions.
    """

    def __init__(
        self,
        pixel_hrzt_count=1338, pixel_hrzt_size=6.45e-6,
        pixel_vert_count=1038, pixel_vert_size=6.45e-6,
        pixel_hrzt_offset=0, pixel_vert_offset=0,
        format_color=DRV_CAM.FORMAT_COLOR.GS, channel_bitdepth=8,
        exposure_mode=DRV_CAM.EXPOSURE_MODE.MANUAL, exposure_time=1e-3,
        acquisition_frames=0, sensitivity=DRV_CAM.SENSITIVITY.NORMAL,
        cls_name="CamCfg", ll_obj=None, **kwargs
    ):
        if "driver" not in kwargs.keys():
            kwargs["driver"] = DRV_DRIVER.CAM
        super().__init__(cls_name=cls_name, **kwargs)
        if ll_obj is not None:
            ll_obj_dict = dict(ll_obj.__dict__)
            for key in list(ll_obj_dict.keys()):
                if key.startswith("_"):
                    del ll_obj_dict[key]
            self.__dict__.update(ll_obj_dict)
        self.pixel_hrzt_count = pixel_hrzt_count
        self.pixel_hrzt_size = pixel_hrzt_size
        self.pixel_hrzt_offset = pixel_hrzt_offset
        self.pixel_vert_count = pixel_vert_count
        self.pixel_vert_size = pixel_vert_size
        self.pixel_vert_offset = pixel_vert_offset
        self.format_color = format_color
        self.channel_bitdepth = channel_bitdepth
        self.exposure_mode = exposure_mode
        self.exposure_time = exposure_time
        self.acquisition_frames = acquisition_frames
        self.sensitivity = sensitivity

    def get_hl_cfg(self):
        return self




###############################################################################


def get_cam_drv(cfg):
    if cfg.model == drv.DRV_MODEL.ALLIEDVISION_MANTA_G145B_NIR:
        return AlliedVisionMantaG145BNIR(cfg)
    elif cfg.model == drv.DRV_MODEL.VRMAGIC_VRMCX:
        return VRmagicVRmCX(cfg)


class CamDrvBase():

    def __init__(self, cfg):
        super().__init__(cfg=cfg)

    @abc.abstractmethod
    def run(self):
        """
        Start capturing images.
        """

    @abc.abstractmethod
    def stop(self):
        """
        Stop capturing images.
        """

    @abc.abstractmethod
    def grab(self):
        """
        Grab image from camera (wait for next image).

        Returns
        -------
        image : numpy.ndarray(3)
            Image aas numpy array.
            Shape: height x width x channel
        """

    @abc.abstractmethod
    def get(self):
        """
        Get image from internal buffer (previously taken image).

        Returns
        -------
        image : numpy.ndarray(3)
            Image aas numpy array.
            Shape: height x width x channel
        """

    def find_exposure_time(self, max_cutoff=2/3, max_iterations=50):
        """
        Varies the camera's exposure time to maximize image brightness
        while keeping the maximum ADC value below saturation.

        Parameters
        ----------
        max_cutoff : float
            Minimum relative brightness required for the image
            maximum.
        max_iterations : int
            Maximum number of iterations.

        Returns
        -------
        prev_exposure_time : float
            Previous exposure time in seconds (s).
            Allows the calling function to reset the settings.
        """
        prev_exposure_time = self.cfg.exposure_time.val
        max_adc = float(2**self.cfg.channel_bitdepth.val - 1)
        loop, verify = True, False
        it = -1
        while loop:
            it += 1
            im = self.grab()
            while im is None:
                im = self.grab()
            saturation = im.max() / max_adc
            if saturation >= 1:
                self.cfg.exposure_time.write(
                    val=self.cfg.exposure_time.val / 3
                )
                self.process()
                time.sleep(0.15)
                verify = False
            elif saturation < max_cutoff:
                scale = 1 / max_cutoff
                if saturation < 0.05:
                    scale = 3
                elif saturation < 0.85:
                    scale = max_cutoff / saturation
                self.cfg.exposure_time.write(
                    val=self.cfg.exposure_time.val * scale
                )
                self.process()
                time.sleep(0.15)
                verify = False
            else:
                if verify:
                    loop = False
                verify = True
            if it >= max_iterations:
                print("\rFinding exposure time: maximum iterations reached")
                return prev_exposure_time
            print("\rFinding exposure time: {:.0f} µs (sat: {:.3f})       "
                  .format(self.cfg.exposure_time.val * 1e6, saturation),
                  end="")
        print("\r", end="                                                  \r")
        return prev_exposure_time

    # ++++ Write/read methods +++++++++++

    def _write_pixel_hrzt_count(self, value):
        pass

    def _read_pixel_hrzt_count(self):
        pass

    def _write_pixel_hrzt_size(self, value):
        pass

    def _read_pixel_hrzt_size(self):
        pass

    def _write_pixel_hrzt_offset(self, value):
        pass

    def _read_pixel_hrzt_offset(self):
        pass

    def _write_pixel_vert_count(self, value):
        pass

    def _read_pixel_vert_count(self):
        pass

    def _write_pixel_vert_size(self, value):
        pass

    def _read_pixel_vert_size(self):
        pass

    def _write_pixel_vert_offset(self, value):
        pass

    def _read_pixel_vert_offset(self):
        pass

    def _write_format_color(self, value):
        pass

    def _read_format_color(self):
        pass

    def _write_channel_bitdepth(self, value):
        pass

    def _read_channel_bitdepth(self):
        pass

    def _write_exposure_mode(self, value):
        pass

    def _read_exposure_mode(self):
        pass

    def _write_exposure_time(self, value):
        pass

    def _read_exposure_time(self):
        pass

    def _write_acquisition_frames(self, value):
        pass

    def _read_acquisition_frames(self):
        pass

    def _write_sensitivity(self, value):
        pass

    def _read_sensitivity(self):
        pass

    # ++++ Helper methods +++++++++++++++

    @property
    def _numpy_dtype(self):
        if self.cfg.channel_bitdepth == 1:
            return "bool"
        elif self.cfg.channel_bitdepth <= 8:
            return "uint8"
        elif self.cfg.channel_bitdepth <= 16:
            return "uint16"
        elif self.cfg.channel_bitdepth <= 32:
            return "uint32"

    @property
    def _numpy_shape(self):
        MAP = {
            drv.DRV_CAM.FORMAT_COLOR.BW: 1,
            drv.DRV_CAM.FORMAT_COLOR.GS: 1,
            drv.DRV_CAM.FORMAT_COLOR.RGB: 3,
            drv.DRV_CAM.FORMAT_COLOR.RGBA: 4
        }
        return (
            self.cfg.pixel_vert_count.val,
            self.cfg.pixel_hrzt_count.val,
            MAP[self.cfg.format_color.val]
        )

    def _cv_buffer_to_numpy(self, buffer):
        return np.ndarray(
            buffer=buffer,
            dtype=self._numpy_dtype,
            shape=self._numpy_shape
        )

    def _callback_delegate(self, buffer):
        self._callback(self._cv_buffer_to_numpy(buffer))

