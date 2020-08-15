from . import alpV42 as alp42

import ctypes as ct
import os

from libics.env import logging
from libics.core.util import misc
from libics.driver.device import DevProperties, STATUS
from libics.driver.interface import ItfBase
from libics.driver.display import Display, FORMAT_COLOR


###############################################################################
# Interface
###############################################################################


class ItfAlp(ItfBase):

    # ALP API hooks
    _alp_itf = None
    # References to ALP API
    _alp_itf_refs = 0
    # ALP device references
    _alp_dev_refs = {}

    LOGGER = logging.get_logger("libics.driver.display.vialux.ItfAlp")

    def __init__(self):
        super().__init__()
        self._is_set_up = False
        self._dev_id = None

    # ++++++++++++++++++++++++++++++++++++++++
    # Device methods
    # ++++++++++++++++++++++++++++++++++++++++

    def setup(self):
        if not self.is_set_up():
            file_dir = os.path.dirname(os.path.realpath(__file__))
            self._alp_itf = alp42.PY_ALP_API(
                dllPath=os.path.join(file_dir, "alpV42.dll")
            )
        self._alp_itf_refs += 1

    def shutdown(self):
        if self.is_set_up():
            self._alp_itf_refs -= 1

    def is_set_up(self):
        return self._alp_itf_refs > 0

    def connect(self, id):
        """
        Raises
        ------
        RuntimeError
            If `id` is not available.
        """
        if self.is_connected():
            if self._dev_id == id:
                return
            else:
                self.close()
        # Check if requested device ID is discovered
        if id not in self._alp_dev_refs:
            self.discover()
            if id not in self._alp_dev_refs:
                raise RuntimeError("device ID unavailable ({:s})".format(id))
        # Set device ID of this interface instance
        self._dev_id = id
        # Check if another interface instance has already opened
        if not self.is_connected():
            ret, _ = self._alp_itf.AlpDevAlloc(int(self._dev_id))
            if ret != alp42.ALP_ERRORS["ALP_OK"]:
                err_msg = "ALP error: {:s}".format(alp42.ErrorsFlipped[ret])
                self.last_error = STATUS(
                    state=STATUS.ERROR, err_type=STATUS.ERR_DEVICE, msg=err_msg
                )
                raise RuntimeError(err_msg)
        self._alp_dev_refs[self._dev_id] += 1

    def close(self):
        self._alp_dev_refs[self._dev_id] -= 1
        if self._alp_dev_refs[self._dev_id] == 0:
            self._alp_itf.AlpDevHalt(self._dev_id)
            self._alp_itf.AlpDevFree(self._dev_id)

    def is_connected(self):
        """
        Raises
        ------
        RuntimeError
            If internal device reference error occured.
        """
        if self._dev_id is None:
            return False
        try:
            # If not discovered
            if self._dev_id not in self._alp_dev_refs:
                return False
            # If discovered but not connected
            elif self._alp_dev_refs[self._dev_id] == 0:
                return False
            # If discovered and connected
            elif self._alp_dev_refs[self._dev_id] > 0:
                return True
            # Handle error
            else:
                assert(False)
        except AssertionError:
            err_msg = "device reference count error"
            self.last_status = STATUS(
                state=STATUS.CRITICAL, err_type=STATUS.ERR_INSTANCE,
                msg=err_msg
            )
            raise RuntimeError(err_msg)

    # ++++++++++++++++++++++++++++++++++++++++
    # Interface methods
    # ++++++++++++++++++++++++++++++++++++++++

    @classmethod
    def discover(cls):
        dev_ids = []
        while True:
            ret, dev_id = cls._alp_itf.AlpDevAlloc()
            if ret == alp42.ALP_ERRORS["ALP_NOT_ONLINE"]:
                break
            dev_ids.append(str(dev_id))
            if ret == alp42.ALP_ERRORS["ALP_OK"]:
                cls._alp_itf.AlpDevFree(dev_id)
        # Check for devices which have become unavailable
        for id in list(cls._alp_dev_refs):
            if id not in dev_ids:
                del cls._alp_dev_refs[id]
                if id in cls._alp_dev_handles:
                    cls.LOGGER.critical(
                        "device lost connection ({:s})".format(id)
                    )
                    del cls._alp_dev_handles[id]
                    # TODO: notify affected devices
        # Check for devices which have been added
        for id in dev_ids:
            if id not in cls._alp_dev_refs:
                cls._alp_dev_refs[id] = 0
        return dev_ids


###############################################################################
# Device
###############################################################################


class VialuxDLP(Display):

    def __init__(self):
        super().__init__()
        self._alp_seq_handle = None
        self.properties.set_properties(**self._get_default_properties_dict(
            "device_name", "temperature"
        ))
        self.seq_properties = DevProperties()
        self.seq_properties.set_device(self)
        self.seq_properties.set_properties(**self._get_default_properties_dict(
            "picture_time", "dark_time", "sequence_repetitions"
        ))

    @property
    def p_seq(self):
        return self.seq_properties

    # ++++++++++++++++++++++++++++++++++++++++
    # Device methods
    # ++++++++++++++++++++++++++++++++++++++++

    def setup(self):
        if not isinstance(self.interface, ItfAlp):
            self.interface = ItfAlp()
        self.interface.setup()

    def shutdown(self):
        if self.is_connected():
            self.close()
        self.interface.shutdown()

    def is_set_up(self):
        return self.interface.is_set_up()

    def connect(self):
        if not self.is_set_up():
            raise RuntimeError("device not set up")
        self.interface.connect(self.identifier)
        self.interface.register(self.identifier, self)
        self.read_device_name()
        self.p.read_all()

    def close(self):
        self.interface.deregister(id=self.identifier)
        self.interface.close()

    def is_connected(self):
        return self.interface.is_connected()

    # ++++++++++++++++++++++++++++++++++++++++
    # Display methods
    # ++++++++++++++++++++++++++++++++++++++++

    def run(self, images=None):
        super().run(images=images, blocking=True)

    def stop(self):
        self._end_displaying()

    def _start_displaying(self):
        # Set up sequence
        seq_length = len(self._images)
        ret, self._alp_seq_handle = self._alp_itf.AlpSeqAlloc(
            self._alp_dev_handle, bitplanes=self.p.channel_bitdepth,
            picnum=seq_length
        )
        if ret != alp42.ALP_ERRORS["ALP_OK"]:
            err_msg = "ALP error: {:s}".format(alp42.ErrorsFlipped[ret])
            self.last_status = STATUS(
                state=STATUS.CRITICAL, err_type=STATUS.ERR_DEVICE, msg=err_msg
            )
            raise RuntimeError(err_msg)
        for i, im in enumerate(self._images):
            im = im.T.copy()
            im = ct.create_string_buffer(im.flatten().tostring())
            self._alp_itf.AlpSeqPut(
                self._alp_dev_handle, self._alp_seq_handle, im,
                picoffset=i, picload=1
            )
        self.p_seq.apply()
        # Start projection
        if self.p.sequence_repetitions == 0:
            self._alp_itf.AlpProjStartCont(
                self._alp_dev_handle, self._alp_seq_handle
            )
        else:
            self._alp_itf.AlpProjStart(
                self._alp_dev_handle, self._alp_seq_handle
            )

    def _end_displaying(self):
        # Stop projection
        self._alp_itf.AlpProjHalt(self._alp_dev_handle)
        # Free sequence
        if self._alp_seq_handle is not None:
            self._alp.AlpSeqFree(self._alp_dev_handle, self._alp_seq_handle)
        self._alp_seq_handle = None

    def is_running(self):
        raise NotImplementedError

    def join_sequence(self):
        """
        Waits for a sequence to finish and returns.

        If continuous projection is set, immediately stops projection.
        """
        if self.is_running():
            if self.p.sequence_repetitions == 0:
                self.stop()
            else:
                self._alp_itf.AlpProjWait(self._alp_dev_handle)

    # ++++++++++++++++++++++++++++++++++++++++
    # Helper methods
    # ++++++++++++++++++++++++++++++++++++++++

    @property
    def _alp_itf(self):
        return self.interface._alp_itf

    @property
    def _alp_dev_handle(self):
        return int(self._dev_id)

    def _seq_is_set_up(self):
        return self._alp_seq_handle is not None

    def _cv_image_bw(self, image):
        """
        Converts an image to 8-bit black/white.

        Parameters
        ----------
        image : `np.ndarray(2)`
            Image to be converted into B/W.

        Returns
        -------
        image : `np.ndarray(2, uint8)`
            Converted image.
        """
        # Check correct size
        target_shape = (self.p.pixel_hrzt_count, self.p.pixel_vert_count)
        image = misc.resize_numpy_array(
            image, target_shape, fill_value=0, mode_keep="front"
        )
        # Apply channel bitdepth
        image[image >= 1] = 1
        image[image < 0] = 0
        image *= 255    # only 1bit encoded in 8bit supported
        return image.astype("uint8")

    # ++++++++++++++++++++++++++++++++++++++++
    # Properties methods
    # ++++++++++++++++++++++++++++++++++++++++

    def read_pixel_hrzt_count(self):
        _, value = self._alp_itf.AlpDevInquire(
            self._alp_dev_handle,
            alp42.ALP_PARMS_DEV_INQUIRE["ALP_DEV_DISPLAY_WIDTH"]
        )
        self.p.pixel_hrzt_count = value
        return value

    def write_pixel_hrzt_count(self, value):
        if value != self.p.pixel_hrzt_count:
            self.LOGGER.warning("cannot write pixel_hrzt_count")

    def read_pixel_hrzt_size(self):
        value = self.read_device_name()
        MAP = {
            "XGA": 13.7e-6, "SXGA_PLUS": 13.7e-6, "1080P_095A": 10.8e-6,
            "XGA_07A": 13.7e-6, "XGA_055A": 10.8e-6, "XGA_055X": 10.8e-6,
            "WUXGA_096A": 10.8e-6,
        }
        value = MAP[value]
        self.p.pixel_hrzt_size = value
        return value

    def write_pixel_hrzt_size(self, value):
        if value != self.p.pixel_hrzt_size:
            self.LOGGER.warning("cannot write pixel_hrzt_size")

    def read_pixel_hrzt_offset(self):
        value = 0
        self.p.pixel_hrzt_offset = 0
        return value

    def write_pixel_hrzt_offset(self, value):
        if value != self.p.pixel_hrzt_offset:
            self.LOGGER.warning("cannot write pixel_hrzt_offset")

    def read_pixel_vert_count(self):
        _, value = self._alp_itf.AlpDevInquire(
            self._alp_dev_handle,
            alp42.ALP_PARMS_DEV_INQUIRE["ALP_DEV_DISPLAY_HEIGHT"]
        )
        self.p.pixel_vert_count = value
        return value

    def write_pixel_vert_count(self, value):
        if value != self.p.pixel_vert_count:
            self.LOGGER.warning("cannot write pixel_vert_count")

    def read_pixel_vert_size(self):
        value = self.read_device_name()
        MAP = {
            "XGA": 13.7e-6, "SXGA_PLUS": 13.7e-6, "1080P_095A": 10.8e-6,
            "XGA_07A": 13.7e-6, "XGA_055A": 10.8e-6, "XGA_055X": 10.8e-6,
            "WUXGA_096A": 10.8e-6,
        }
        value = MAP[value]
        self.p.pixel_vert_size = value
        return value

    def write_pixel_vert_size(self, value):
        if value != self.p.pixel_vert_size:
            self.LOGGER.warning("cannot write pixel_vert_size")

    def read_pixel_vert_offset(self):
        value = 0
        self.p.pixel_vert_offset = 0
        return value

    def write_pixel_vert_offset(self, value):
        if value != self.p.pixel_vert_offset:
            self.LOGGER.warning("cannot write pixel_vert_offset")

    def read_format_color(self):
        if self.p.format_color is None:
            self.p.format_color = FORMAT_COLOR.BW
        return self.p.format_color

    def write_format_color(self, value):
        # TODO: add support for FORMAT_COLOR.GS
        if value != FORMAT_COLOR.GS:
            raise NotImplementedError
        self.p.format_color = value

    def read_channel_bitdepth(self):
        if self.p.channel_bitdepth is None:
            self.p.channel_bitdepth = 1
        return self.p.channel_bitdepth

    def write_channel_bitdepth(self, value):
        # TODO: add support for >1
        if value != 1:
            raise NotImplementedError
        self.p.channel_bitdepth = value

    def read_picture_time(self):
        if self._seq_is_set_up():
            _, value = self._alp_itf.AlpSeqInquire(
                self._alp_dev_handle, self._alp_seq_handle,
                alp42.ALP_PARMS_SEQ_INQUIRE["ALP_PICTURE_TIME"]
            )
            value = value * 1e-6
            self.p.picture_time = value
        else:
            value = self.p.picture_time
        return value

    def _write_picture_time(self, value):
        if self._seq_is_set_up():
            picture_time = int(round(1e6 * value))
            illuminate_time = int(round(1e6 * (value - self.p.dark_time)))
            self._alp_itf.AlpSeqTiming(
                self._alp_dev_handle, self._alp_seq_handle,
                illuminatetime=illuminate_time, picturetime=picture_time
            )
        self.p.picture_time = value

    def read_dark_time(self):
        if self._seq_is_set_up():
            _, m = self._alp_itf.AlpSeqInquire(
                self._alp_dev_handle, self._alp_seq_handle,
                alp42.ALP_PARMS_SEQ_INQUIRE["ALP_BIN_MODE"]
            )
            if m == alp42.ALP_PARMS_SEQ_CONTROL_VALUE["ALP_BIN_UNINTERRUPTED"]:
                value = 0
            else:
                picture_time = self.read_picture_time()
                _, value = self._alp_itf.AlpSeqInquire(
                    self._alp_dev_handle, self._alp_seq_handle,
                    alp42.ALP_PARMS_SEQ_INQUIRE["ALP_ILLUMINATE_TIME"]
                )
                illuminate_time = 1e-6 * value
                value = picture_time - illuminate_time
        else:
            value = self.p.dark_time
        self.p.dark_time = value
        return value

    def write_dark_time(self, value):
        if self._seq_is_set_up():
            if value == 0:
                self._alp_itf.AlpSeqControl(
                    self._alp_dev_handle, self._alp_seq_handle,
                    alp42.ALP_PARMS_SEQ_CONTROL_TYPE["ALP_BIN_MODE"],
                    alp42.ALP_PARMS_SEQ_CONTROL_VALUE["ALP_BIN_UNINTERRUPTED"]
                )
            else:
                self._alp_itf.AlpSeqControl(
                    self._alp_dev_handle, self._alp_seq_handle,
                    alp42.ALP_PARMS_SEQ_CONTROL_TYPE["ALP_BIN_MODE"],
                    alp42.ALP_PARMS_SEQ_CONTROL_VALUE["ALP_BIN_NORMAL"]
                )
            picture_time = int(round(1e6 * self.p.picture_time))
            illuminate_time = int(round(1e6 * (self.p.picture_time - value)))
            self._alp_itf.AlpSeqTiming(
                self._alp_dev_handle, self._alp_seq_handle,
                illuminatetime=illuminate_time, picturetime=picture_time
            )
        self.p.picture_time = value

    def read_sequence_repetitions(self):
        if self.p.sequence_repetitions == 0:
            return 0
        elif self._seq_is_set_up():
            _, value = self._alp_itf.AlpSeqInquire(
                self._alp_dev_handle, self._alp_seq_handle,
                alp42.ALP_PARMS_SEQ_INQUIRE["ALP_SEQ_REPEAT"]
            )
            self.p.sequence_repetitions = value
            return value
        else:
            return self.p.sequence_repetitions

    def write_sequence_repetitions(self, value):
        if self._seq_is_set_up() and value > 0:
            self._alp_itf.AlpSeqControl(
                self._alp_dev_handle, self._alp_seq_handle,
                alp42.ALP_PARMS_SEQ_CONTROL_TYPE["ALP_SEQ_REPEAT"],
                max(1, min(1048576, int(round(value))))
            )
        self.p.sequence_repetitions = value

    def read_device_name(self):
        _, value = self._alp_itf.AlpDevInquire(
            self._alp_dev_handle,
            alp42.ALP_PARMS_DEV_INQUIRE["ALP_DEV_DMDTYPE"]
        )
        CTRL_VAL = alp42.ALP_PARMS_DEV_CONTROL_VALUE
        MAP = {
            CTRL_VAL["ALP_DMD_TYPE_XGA"]: "XGA",
            CTRL_VAL["ALP_DMD_TYPE_SXGA_PLUS"]: "SXGA_PLUS",
            CTRL_VAL["ALP_DMD_TYPE_1080P_095A"]: "1080P_095A",
            CTRL_VAL["ALP_DMD_TYPE_XGA_07A"]: "XGA_07A",
            CTRL_VAL["ALP_DMD_TYPE_XGA_055A"]: "XGA_055A",
            CTRL_VAL["ALP_DMD_TYPE_XGA_055X"]: "XGA_055X",
            CTRL_VAL["ALP_DMD_TYPE_WUXGA_096A"]: "WUXGA_096A",
        }
        value = MAP[value]
        self.p.device_name = value
        return value

    def write_device_name(self, value):
        self.LOGGER.warning("cannot write device_name")

    def read_temperature(self):
        _, value = self._alp_itf.AlpDevInquire(
            self._alp_dev_handle,
            alp42.ALP_PARMS_DEV_INQUIRE["ALP_DDC_FPGA_TEMPERATURE"]
        )
        value = 256.0 * value
        self.p.temperature = value
        return value

    def write_temperature(self):
        self.LOGGER.warning("cannot write temperature")
