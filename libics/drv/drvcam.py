import abc

import numpy as np

from libics.drv import drv


###############################################################################


def get_cam_drv(cfg):
    if cfg.model == drv.DRV_MODEL.ALLIEDVISION_MANTA_G145B_NIR:
        return AlliedVisionMantaG145BNIR(cfg)


class CamDrvBase(drv.DrvBase):

    def __init__(self, cfg):
        super().__init__(cfg=cfg)

    @abc.abstractmethod
    def grab(self):
        pass

    @abc.abstractmethod
    def run(self):
        pass

    @abc.abstractmethod
    def stop(self):
        pass

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
            self.cfg.pixel_vert_count,
            self.cfg._pixel_hrzt_count,
            MAP[self.cfg.format_color]
        )


###############################################################################


class AlliedVisionMantaG145BNIR(CamDrvBase):

    def __init__(self, cfg):
        super().__init__(cfg)
        self._callback = None

    def grab(self):
        return self._cv_buffer_to_numpy(
            self._interface.grab_data(index=self._interface.latest_index)
        )

    def run(self, callback=None):
        self._callback = callback
        callback = self._callback_delegate if callback is not None else None
        self._interface.setup_frames(callback=callback)
        self._interface.start_capture()
        self._interface.start_acquisition()

    def stop(self):
        self._interface.end_acquisition()
        self._interface.flush_capture_queue()
        self._interface.end_capture()
        self._interface.revoke_all_frames()

    # ++++ Write/read methods +++++++++++

    def _write_pixel_hrzt_count(self, value):
        self._interface.cam.Width = value

    def _read_pixel_hrzt_count(self):
        return self._interface.cam.Width

    def _read_pixel_hrzt_size(self):
        return 6.45e-6

    def _write_pixel_hrzt_offset(self, value):
        self._interface.cam.OffsetX = value

    def _read_pixel_hrzt_offset(self):
        return self._interface.cam.OffsetX

    def _write_pixel_vert_count(self, value):
        self._interface.cam.Height = value

    def _read_pixel_vert_count(self):
        return self._interface.cam.Height

    def _read_pixel_vert_size(self):
        return 6.45e-6

    def _write_pixel_vert_offset(self, value):
        self._interface.cam.OffsetY = value

    def _read_pixel_vert_offset(self):
        return self._interface.cam.OffsetY

    def _read_format_color(self):
        return drv.DRV_CAM.FORMAT_COLOR.GS

    def _write_channel_bitdepth(self, value):
        MAP = {
            8: "Mono8",
            12: "Mono12"
        }
        self._interface.cam.PixelFormat = MAP[value]

    def _read_channel_bitdepth(self):
        MAP = {
            "Mono8": 8,
            "Mono12": 12,
            "Mono12Packed": 12
        }
        return MAP[self._interface.cam.PixelFormat]

    def _write_exposure_mode(self, value):
        MAP = {
            drv.DRV_CAM.EXPOSURE_MODE.MANUAL: "Off",
            drv.DRV_CAM.EXPOSURE_MODE.CONTINUOS: "Continuous",
            drv.DRV_CAM.EXPOSURE_MODE.SINGLE: "Single"
        }
        self._interface.cam.ExposureAuto = MAP[value]

    def _read_exposure_mode(self):
        MAP = {
            "Off": drv.DRV_CAM.EXPOSURE_MODE.MANUAL,
            "Continuous": drv.DRV_CAM.EXPOSURE_MODE.CONTINUOS,
            "Single": drv.DRV_CAM.EXPOSURE_MODE.SINGLE
        }
        return MAP[self._interface.cam.ExposureAuto]

    def _write_exposure_time(self, value):
        self._interface.cam.ExposureTimeAbs = 1e6 * value

    def _read_exposure_time(self):
        return self._interface.cam.ExposureTimeAbs / 1e6

    def _write_acquisition_frames(self, value):
        if value == 0:
            self._interface.cam.AcquisitionMode = "Continuous"
        elif value == 1:
            self._interface.cam.AcquisitionMode = "SingleFrame"
        else:
            self._interface.cam.AcquisitionMode = "MultiFrame"
            self._interface.cam.AcquisitionFrameCount = value

    def _read_acquisition_frames(self):
        value = self._interface.cam.AcquisitionMode
        MAP = {
            "Continuous": 0,
            "SingleFrame": 1,
            "MultiFrame": self._interface.cam.AcquisitionFrameCount
        }
        return MAP[value]

    def _write_sensitivity(self, value):
        MAP = {
            drv.DRV_CAM.SENSITIVITY.NORMAL: "Off",
            drv.DRV_CAM.SENSITIVITY.NIR_FAST: "On_Fast",
            drv.DRV_CAM.SENSITIVITY.NIR_HQ: "On_HighQuality",
        }
        self._interface.cam.NirMode = MAP[value]

    def _read_sensitivity(self):
        MAP = {
            "Off": drv.DRV_CAM.SENSITIVITY.NORMAL,
            "On_Fast": drv.DRV_CAM.SENSITIVITY.NIR_FAST,
            "On_HighQuality": drv.DRV_CAM.SENSITIVITY.NIR_HQ
        }
        return MAP[self._interface.cam.NirMode]

    # ++++ Helper methods +++++++++++++++

    def _cv_buffer_to_numpy(self, buffer):
        return np.ndarray(
            buffer=buffer,
            dtype=self._numpy_dtype,
            shape=self._numpy_shape
        )

    def _callback_delegate(self, buffer):
        self._callback(self._cv_buffer_to_numpy(buffer))
