import abc
import time

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


###############################################################################


class AlliedVisionMantaG145BNIR(CamDrvBase):

    def __init__(self, cfg):
        super().__init__(cfg)
        self._callback = None

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

    def grab(self):
        _index = self._interface.next_index
        if not self._interface.cfg.frame_requeue:
            self._interface.queue_frame_capture(index=_index)
        if self._interface.wait_frame_capture(index=_index) == 0:
            return self._cv_buffer_to_numpy(
                self._interface.grab_data(index=_index)
            )

    def get(self):
        return self._cv_buffer_to_numpy(
            self._interface.grab_data(index=self._interface.latest_index)
        )

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
