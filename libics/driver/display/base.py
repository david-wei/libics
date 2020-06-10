
class DRV_DSP:

    class FORMAT_COLOR:

        BW = 0
        GS = 1
        RGB = 2
        RGBA = 3


class DspCfg():

    """
    DrvCfgBase -> DspCfg.

    Parameters
    ----------
    pixel_hrzt_count, pixel_vert_count : int
        Pixel count in respective direction.
    pixel_hrzt_size, pixel_vert_size : float
        Pixel size in meters in respective direction.
    pixel_hrzt_offset, pixel_vert_offset : int
        Offset of pixels to be captured.
    format_color : DRV_DSP.FORMAT_COLOR
        BW: black/white boolean image.
        GS: greyscale image.
        RGB: RGB color image.
        RGBA: RGB image with alpha channel.
    channel_bitdepth : int
        Bits per color channel.
    picture_time : float
        Time in seconds (s) each single image is shown.
    dark_time : float
        Time in seconds (s) between images in a sequence.
    sequence_repetitions : int
        Number of sequences to be shown.
        0 (zero) is interpreted as infinite, i.e.
        continuos repetition.
    temperature : float
        Display temperature in Celsius (°C).
    """

    def __init__(
        self,
        pixel_hrzt_count=1024, pixel_hrzt_size=13.68e-6,
        pixel_vert_count=768, pixel_vert_size=13.68e-6,
        pixel_hrzt_offset=0, pixel_vert_offset=0,
        format_color=DRV_DSP.FORMAT_COLOR.GS, channel_bitdepth=8,
        picture_time=9.0, dark_time=0.0, sequence_repetitions=0,
        temperature=25.0,
        cls_name="DspCfg", ll_obj=None, **kwargs
    ):
        if "driver" not in kwargs.keys():
            kwargs["driver"] = DRV_DRIVER.DSP
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
        self.picture_time = picture_time
        self.dark_time = dark_time
        self.sequence_repetitions = sequence_repetitions
        self.temperature = temperature

    def get_hl_cfg(self):
        return self




###############################################################################


def get_dsp_drv(cfg):
    if cfg.model == drv.DRV_MODEL.TEXASINSTRUMENTS_DLP7000:
        return TexasInstrumentsDLP7000(cfg)


class DspDrvBase():

    def __init__(self, cfg):
        super().__init__(cfg=cfg)

    @abc.abstractmethod
    def run(self, images):
        """
        Show an image sequence on display.
        """

    @abc.abstractmethod
    def stop(self, images):
        """
        Stops displaying the image sequence.
        """

    # ++++ Write/read methods +++++++++++

    def _write_pixel_hrzt_count(self, value):
        pass

    def _read_pixel_hrzt_count(self):
        return self.cfg.pixel_hrzt_count.val

    def _write_pixel_hrzt_size(self, value):
        pass

    def _read_pixel_hrzt_size(self):
        return self.cfg.pixel_hrzt_size.val

    def _write_pixel_hrzt_offset(self, value):
        pass

    def _read_pixel_hrzt_offset(self):
        return self.cfg.pixel_hrzt_offset.val

    def _write_pixel_vert_count(self, value):
        pass

    def _read_pixel_vert_count(self):
        return self.cfg.pixel_vert_count.val

    def _write_pixel_vert_size(self, value):
        pass

    def _read_pixel_vert_size(self):
        return self.cfg.pixel_vert_size.val

    def _write_pixel_vert_offset(self, value):
        pass

    def _read_pixel_vert_offset(self):
        return self.cfg.pixel_vert_offset.val

    def _write_format_color(self, value):
        pass

    def _read_format_color(self):
        return self.cfg.format_color.val

    def _write_channel_bitdepth(self, value):
        pass

    def _read_channel_bitdepth(self):
        return self.cfg.channel_bitdepth.val

    def _write_picture_time(self, value):
        pass

    def _read_picture_time(self):
        pass

    def _write_dark_time(self, value):
        pass

    def _read_dark_time(self):
        pass

    def _write_sequence_repetitions(self, value):
        pass

    def _read_sequence_repetitions(self):
        pass

    def _write_temperature(self, value):
        pass

    def _read_temperature(self, value):
        pass
