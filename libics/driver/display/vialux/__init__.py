import os

from libics.core import env
from . import alpV42 as alp42


###############################################################################
# Initialization
###############################################################################


# Global variable for Vialux ALP4.2 API object
_ALP42 = None


def startup_alp42():
    """
    Initializes the Vialux ALP4.2 C API.

    The Vialux ALP4.2 API requires a startup and a shutdown call.
    This function checks whether startup has already been called.

    Returns
    -------
    _ALP42 : alpV42.PY_ALP_API
        Vialux ALP4.2 API object.
    """
    global _ALP42
    if _ALP42 is None:
        _ALP42 = alp42.PY_ALP_API(
            dllPath=os.path.join(env.DIR_ITFAPI, "alpV42.dll")
        )
    return _ALP42




###############################################################################


class VialuxItf():

    def __init__(self, cfg):
        super().__init__(cfg)
        self._dmd = None
        self._alp = None
        self._seq = None
        self._seq_repetitions = None
        self.API = None

    def setup(self):
        self._alp = vialux.startup_alp42()
        self.API = vialux.alp42
        if self.cfg.device is None:
            self.cfg.device = 0
        elif isinstance(self.cfg.device, str):
            self.cfg.device = int(float(self.cfg.device))
        _, self._dmd = self._alp.AlpDevAlloc(self.cfg.device)

    def shutdown(self):
        self._alp.AlpDevHalt(self._dmd)
        self._alp.AlpDevFree(self._dmd)
        self._dmd = None
        self._alp = None
        self.API = None

    def connect(self):
        pass

    def close(self):
        pass

    def init(self, bitdepth, sequence_length):
        """
        Initializes a sequence by allocating memory for image display.

        Parameters
        ----------
        bitdepth : int
            Channel bitdepth of image.
        sequence_length : int or list
            Number of images in sequence or sequence itself.
        """
        bitdepth = 1    # only 1 possible for current implementation
        if hasattr(sequence_length, "__len__"):
            sequence_length = len(sequence_length)
        _, self._seq = self._alp.AlpSeqAlloc(
            self._dmd, bitplanes=bitdepth, picnum=sequence_length
        )

    def run(self, sequence, bitdepth, repetitions):
        """
        Loads the sequence memory and starts displaying the images.

        Parameters
        ----------
        sequence : list(np.ndarray(2))
            List of images.
        bitdepth : int
            Channel bitdepth of image.
        repetitions : int
            Number of repetitions of the sequence.
            0 (zero) is interpreted as continuos display.

        Notes
        -----
        As convenience function, one can directly run a sequence
        without init, but no sequence control modifications can
        be made.
        """
        if self._seq is None:
            self.init(bitdepth, len(sequence))
        self._seq_repetitions = repetitions
        for i, image in enumerate(sequence):
            image = image.T.copy()
            image = ctypes.create_string_buffer(image.flatten().tostring())
            self._alp.AlpSeqPut(
                self._dmd, self._seq, image, picoffset=i, picload=1
            )
        if self._seq_repetitions == 0:
            self._alp.AlpProjStartCont(self._dmd, self._seq)
        else:
            self._alp.AlpProjStart(self._dmd, self._seq)

    def stop(self):
        """
        Stops any playing sequence.
        """
        self._alp.AlpProjHalt(self._dmd)
        if self._seq is not None:
            self._alp.AlpSeqFree(self._dmd, self._seq)
        self._alp.AlpDevHalt(self._dmd)
        _, self._dmd = self._alp.AlpDevAlloc(self.cfg.device)
        self._seq = None
        self._seq_repetitions = None

    # ++++ Wrapper methods ++++++++++++++

    def dev_inq(self, key):
        """Device inquiry."""
        return self._alp.AlpDevInquire(self._dmd, key)[1]

    def dev_ctrl(self, key, val):
        """Device control."""
        return self._alp.AlpDevControl(self._dmd, key, val)

    def seq_inq(self, key):
        """Sequence inquiry."""
        if self._seq is None:
            raise err.RUNTM_DRV_DSP(
                err.RUNTM_DRV_DSP.str("no sequence allocated")
            )
        else:
            return self._alp.AlpSeqInquire(self._dmd, self._seq, key)[1]

    def seq_ctrl(self, key, val):
        """Sequence control."""
        if self._seq is not None:
            return self._alp.AlpSeqControl(self._dmd, self._seq, key, val)

    def seq_time(self, **kwargs):
        """
        Set sequence timing.

        Parameters
        ----------
        illuminatetime : int
            Illuminate time in microseconds (µs).
        picturetime : int
            Picture time in microseconds (µs).
        """
        self._alp.AlpSeqTiming(self._dmd, self._seq, **kwargs)

    def seq_wait(self):
        """
        Waits for a sequence to finish and returns.

        Returns
        -------
        wait_success : bool or None
            True: Sequence has finished.
            False: Sequence in continuos mode, cannot finish.
            None: No sequence is running.
        """
        if self._seq_repetitions is None:
            return None
        elif self._seq_repetitions == 0:
            return False
        else:
            self._alp.AlpProjHalt(self._dmd)
            return True


class BinVialuxCfg():

    """
    ProtocolCfgBase -> BinCfgBase -> BinVialuxCfg.

    Parameters
    ----------
    """

    def __init__(self,
                 cls_name="BinVialuxCfg", ll_obj=None, **kwargs):
        if "interface" not in kwargs.keys():
            kwargs["interface"] = ITF_BIN.VIALUX
        super().__init__(cls_name=cls_name, **kwargs)
        if ll_obj is not None:
            self.__dict__.update(ll_obj.__dict__)

    def get_hl_cfg(self):
        return self











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
                self._interface.API.ALP_PARMS_SEQ_INQUIRE["ALP_SEQ_REPEAT"]
            )
        return sequence_repetitions

    def _read_temperature(self):
        return 256.0 * self._interface.dev_inq(
            self._interface.API
            .ALP_PARMS_DEV_INQUIRE["ALP_DDC_FPGA_TEMPERATURE"]
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
