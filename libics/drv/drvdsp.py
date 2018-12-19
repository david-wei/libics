import abc

import numpy as np

from libics.drv import drv
from libics.util import misc


###############################################################################


def get_dsp_drv(cfg):
    if cfg.model == drv.DRV_MODEL.TEXASINSTRUMENTS_DLP7000:
        return TexasInstrumentsDLP7000(cfg)


class DspDrvBase(drv.DrvBase):

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


###############################################################################


class TexasInstrumentsDLP7000(DspDrvBase):

    def __init__(self, cfg):
        super().__init__(cfg)

    def init(self, images):
        """
        Allocates memory in the DMD.

        Parameters
        ----------
        images : list(np.ndarray(2, float)) or np.ndarray(2, float)
            (List of) images to be displayed. Greyscales are
            normalized to [0, 1] where values outside the bounds
            are changed to the respective bound value.

        """
        if not isinstance(images, list):
            images = [images]
        self.__images = [self._cv_image(image) for image in images]
        self._interface.init(self.cfg.channel_bitdepth.val, len(self.__images))

    def run(self, images=None):
        """
        Displays the given images on the DMD. Note that the images are expected
        to be normalized to the interval [0, 1] corresponding to static on/off
        states. Intermediate values request time modulation with picture times
        as set by the configuration.

        Parameters
        ----------
        images : list(np.ndarray(2, float)) or np.ndarray(2, float) or None
            (List of) images to be displayed. Greyscales are
            normalized to [0, 1] where values outside the bounds
            are changed to the respective bound value.
            None: Uses images set by the init method.

        Notes
        -----
        TODO: Implement grayscale images. Currently only boolean bitdepth (1)
              is used.
        """
        if images is not None:
            if not isinstance(images, list):
                images = [images]
            self.__images = [self._cv_image(image) for image in images]
        self._interface.run(
            self.__images, self.cfg.channel_bitdepth.val,
            self.cfg.sequence_repetitions.val
        )

    def stop(self):
        self._interface.stop()

    # ++++ Write/read methods +++++++++++

    def _read_pixel_hrzt_count(self):
        return 1024

    def _read_pixel_hrzt_size(self):
        return 13.68e-6

    def _read_pixel_hrzt_offset(self):
        return 0

    def _read_pixel_vert_count(self):
        return 768

    def _read_pixel_vert_size(self):
        return 13.68e-6

    def _read_pixel_vert_offset(self):
        return 0

    def _read_format_color(self):
        return drv.DRV_DSP.FORMAT_COLOR.GS

    def _write_picture_time(self, value):
        picture_time = value
        illuminate_time = self.cfg.picture_time.val - self.cfg.dark_time.val
        self._interface.seq_time(
            picturetime=int(round(picture_time * 1e6)),
            illuminatetime=int(round(illuminate_time * 1e6))
        )

    def _read_picture_time(self):
        return 1e-6 * self._interface.seq_inq(
            self._interface.API.ALP_PARMS_SEQ_INQUIRE["ALP_PICTURE_TIME"]
        )

    def _write_dark_time(self, value):
        if value == 0:
            self._interface.seq_ctrl(
                self._interface.API.ALP_PARMS_SEQ_CONTROL_TYPE["ALP_BIN_MODE"],
                self._interface.API
                .ALP_PARMS_SEQ_CONTROL_VALUE["ALP_BIN_UNINTERRUPTED"]
            )
        else:
            self._interface.seq_ctrl(
                self._interface.API.ALP_PARMS_SEQ_CONTROL_TYPE["ALP_BIN_MODE"],
                self._interface.API
                .ALP_PARMS_SEQ_CONTROL_VALUE["ALP_BIN_NORMAL"]
            )
        picture_time = self.cfg.picture_time.val
        illuminate_time = picture_time - value
        self._interface.seq_time(
            picturetime=int(round(picture_time * 1e6)),
            illuminatetime=int(round(illuminate_time * 1e6))
        )

    def _read_dark_time(self):
        uninterrupted = (
            self._interface.seq_inq(
                self._interface.API.ALP_PARMS_SEQ_INQUIRE["ALP_BIN_MODE"]
            ) == self._interface
            .API.ALP_PARMS_SEQ_CONTROL_VALUE["ALP_BIN_UNINTERRUPTED"]
        )
        if uninterrupted:
            return 0
        else:
            picture_time = 1e-6 * self._interface.seq_inq(
                self._interface.API.ALP_PARMS_SEQ_INQUIRE["ALP_PICTURE_TIME"]
            )
            illuminate_time = 1e-6 * self._interface.seq_inq(
                self._interface.API
                .ALP_PARMS_SEQ_INQUIRE["ALP_ILLUMINATE_TIME"]
            )
            return picture_time - illuminate_time

    def _write_sequence_repetitions(self, value):
        if value > 0:
            self._interface.seq_ctrl(
                self._interface.API
                .ALP_PARMS_SEQ_CONTROL_TYPE["ALP_SEQ_REPEAT"],
                max(1, min(1048576, int(round(value))))
            )

    def _read_sequence_repetitions(self):
        sequence_repetitions = self.cfg.sequence_repetitions.val
        if sequence_repetitions > 0:
            sequence_repetitions = self._interface.seq_inq(
                self._interface.API.ALP_PARAMS_SEQ_INQUIRE["ALP_SEQ_REPEAT"]
            )
        return sequence_repetitions

    def _read_temperature(self, value):
        return 256.0 * self._interface.dev_inq(
            self._interface.API
            .ALP_PARAMS_DEV_INQUIRE["ALP_DDC_FPGA_TEMPERATURE"]
        )

    # ++++ Helper methods +++++++++++++++

    def _cv_image(self, image):
        # Convert image to 2D greyscale array
        image = np.array(image, dtype=float)
        if len(image.shape) == 3:
            image = np.mean(image, axis=-1)
            image = np.squeeze(image, axis=-1)
        # Check correct size
        target_shape = (
            self.cfg.pixel_hrzt_count.val, self.cfg.pixel_vert_count.val
        )
        image = misc.resize_numpy_array(
            image, target_shape, fill_value=0, mode_keep="front"
        )
        # Apply channel bitdepth
        image[image > 1] = 1
        image[image < 0] = 0
        bitdepth = self.cfg.channel_bitdepth.val
        bitdepth = 1    # only 1bit currently supported
        image *= (2**bitdepth - 1)
        image *= 255    # only 1bit encoded in 8bit supported
        dtype = None
        if bitdepth <= 8:
            dtype = "uint8"
        elif bitdepth > 8 and bitdepth <= 16:
            dtype = "uint16"
        else:
            dtype = "uint32"
        return image.round().astype(dtype)
