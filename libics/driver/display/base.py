import abc
import threading

from libics.core.util.func import StoppableThread
from libics.driver.device import DevBase


###############################################################################


class FORMAT_COLOR:

    BW = "BW"
    GS = "GS"
    RGB = "RGB"
    RGBA = "RGBA"


###############################################################################


class Display(DevBase):

    """
    Properties
    ----------
    pixel_hrzt_count, pixel_vert_count : `int`
        Pixel count in respective direction.
    pixel_hrzt_size, pixel_vert_size : `float`
        Pixel size in meters in respective direction.
    pixel_hrzt_offset, pixel_vert_offset : `int`
        Offset of pixels to be captured.
    format_color : `FORMAT_COLOR`
        BW: black/white boolean image.
        GS: greyscale image.
        RGB: RGB color image.
        RGBA: RGB image with alpha channel.
    channel_bitdepth : `int`
        Bits per color channel.
    picture_time : `float`
        Time in seconds (s) each single image is shown.
    dark_time : `float`
        Time in seconds (s) between images in a sequence.
    sequence_repetitions : `int`
        Number of sequences to be shown.
        0 (zero) is interpreted as infinite, i.e.
        continuos repetition.
    """
    def __init__(self):
        super().__init__()
        self.properties.set_properties(self._get_default_properties_dict(
            "pixel_hrzt_count", "pixel_vert_count",
            "pixel_hrzt_size", "pixel_vert_size",
            "pixel_hrzt_offset", "pixel_vert_offset",
            "format_color", "channel_bitdepth",
            "picture_time", "dark_time", "sequence_repetitions"
        ))
        self._is_running = threading.Lock()
        self._images = []
        self._display_thread = None

    # ++++++++++++++++++++++++++++++++++++++++
    # Display methods
    # ++++++++++++++++++++++++++++++++++++++++

    def run(self, images=None, blocking=False):
        """
        Show an image sequence on display.

        Parameters
        ----------
        images : `np.ndarray(2)` or `list(np.ndarray(2))`
            Image sequence to be displayed.
            If `None`, uses the previously set images.
        blocking : `bool`
            Flag whether to start displaying in the calling thread.
        """
        if self.is_running():
            self.stop()
        if images is not None:
            self._images = images
        if blocking:
            self._start_displaying()
        else:
            self._display_thread = StoppableThread(
                target=self._start_displaying
            )
            self._display_thread.start()

    def stop(self):
        """
        Stops displaying the image sequence.
        """
        if self.is_running():
            if self._display_thread is not None:
                self._display_thread.stop()
            self._end_displaying()
        self._display_thread = None

    @abc.abstractmethod
    def _start_displaying(self):
        """
        Issues the device to start displaying.

        Notes
        -----
        Lock :py:attr:`_is_running`!
        """

    @abc.abstractmethod
    def _end_displaying(self):
        """
        Issues the device to end displaying.

        Notes
        -----
        Unlock :py:attr:`_is_running`!
        """

    def is_running(self):
        """
        Checks whether camera is capturing.
        """
        return self._is_running.locked()

    # ++++++++++++++++++++++++++++++++++++++++
    # Properties methods
    # ++++++++++++++++++++++++++++++++++++++++

    @abc.abstractmethod
    def read_pixel_hrzt_count(self):
        pass

    @abc.abstractmethod
    def write_pixel_hrzt_count(self, value):
        pass

    @abc.abstractmethod
    def read_pixel_hrzt_size(self):
        pass

    @abc.abstractmethod
    def write_pixel_hrzt_size(self, value):
        pass

    @abc.abstractmethod
    def read_pixel_hrzt_offset(self):
        pass

    @abc.abstractmethod
    def write_pixel_hrzt_offset(self, value):
        pass

    @abc.abstractmethod
    def read_pixel_vert_count(self):
        pass

    @abc.abstractmethod
    def write_pixel_vert_count(self, value):
        pass

    @abc.abstractmethod
    def read_pixel_vert_size(self):
        pass

    @abc.abstractmethod
    def write_pixel_vert_size(self, value):
        pass

    @abc.abstractmethod
    def read_pixel_vert_offset(self):
        pass

    @abc.abstractmethod
    def write_pixel_vert_offset(self, value):
        pass

    @abc.abstractmethod
    def read_format_color(self):
        pass

    @abc.abstractmethod
    def write_format_color(self, value):
        pass

    @abc.abstractmethod
    def read_channel_bitdepth(self):
        pass

    @abc.abstractmethod
    def write_channel_bitdepth(self, value):
        pass

    @abc.abstractmethod
    def read_picture_time(self):
        pass

    @abc.abstractmethod
    def write_picture_time(self, value):
        pass

    @abc.abstractmethod
    def read_dark_time(self):
        pass

    @abc.abstractmethod
    def write_dark_time(self, value):
        pass

    @abc.abstractmethod
    def read_sequence_repetitions(self):
        pass

    @abc.abstractmethod
    def write_sequence_repetitions(self, value):
        pass
