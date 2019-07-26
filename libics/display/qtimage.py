# Qt Imports
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QFrame, QLabel, QHBoxLayout, QSizePolicy, QWidget
)


###############################################################################


class QScaledLabel(QLabel, object):

    """
    `QtWidgets.QLabel` where its pixmap is scaled with the label.

    Parameters
    ----------
    aspect_ratio : int or float or None, optional
        `int`, `float`:
            Fixed aspect ratio (width/height) of scaled label.
        `None`:
            Fluent aspect ratio, resizes with label space.
    *args, **kwargs
        Arguments passed to `QLabel` constructor.

    Notes
    -----
    Overloaded methods:
    * `resizeEvent(event)`:
          Calls the custom `setPixmap` method to scale the pixmap.
    * `setPixmap(pixmap)` -> `set_pixmap(pixmap)`:
          Sets the original pixmap ready to be scaled.
    """

    def __init__(self, *args, aspect_ratio=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_scaling(aspect_ratio)
        self._pixmap = None

    @staticmethod
    def _get_image_size(px_width, px_height, aspect_ratio,
                        rect_width, rect_height):
        """
        Calculates the scaled image size given the image size, the
        image's aspect ratio and the given canvas space.
        """
        width, height = float(rect_width), float(rect_height)
        px_width = px_width * aspect_ratio
        scale = (width / px_width, height / px_height)
        if scale[0] > scale[1]:
            width = scale[1] * px_width
        elif scale[0] < scale[1]:
            height = scale[0] * px_height
        return int(round(width)), int(round(height))

    def set_scaling(self, aspect_ratio):
        """
        Sets the `QLabel` scaling behaviour.

        Parameters
        ----------
        aspect_ratio : int or float or None, optional
            `int`, `float`:
                Fixed aspect ratio (width/height) of scaled label.
            `None`:
                Fluent aspect ratio, resizes with label space.
        """
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        if aspect_ratio is None:
            self._aspect_ratio = None
        else:
            self._aspect_ratio = aspect_ratio

    def set_pixmap(self, pixmap):
        """
        Caches the original, unscaled pixmap.
        """
        self._pixmap = pixmap
        self._set_scaled_pixmap()

    def _set_scaled_pixmap(self):
        """
        Uses the cached pixmap to draw a scaled pixmap.
        """
        if self._pixmap is not None:
            if self._aspect_ratio is None:
                super().setPixmap(self._pixmap.scaled(
                    self.width(), self.height(),
                    aspectRatioMode=Qt.IgnoreAspectRatio
                ))
            elif self._aspect_ratio == 1:
                super().setPixmap(self._pixmap.scaled(
                    self.width(), self.height(),
                    aspectRatioMode=Qt.KeepAspectRatio
                ))
            else:
                width, height = QScaledLabel._get_image_size(
                    self._pixmap.width(), self._pixmap.height(),
                    self._aspect_ratio,
                    self.width(), self.height()
                )
                super().setPixmap(self._pixmap.scaled(
                    width, height, aspectRatioMode=Qt.IgnoreAspectRatio
                ))

    def resizeEvent(self, event):
        """
        Scales the pixmap to the new size.
        """
        self._set_scaled_pixmap()
        super().resizeEvent(event)


###############################################################################


class QtImage(QWidget, object):

    """
    Qt widget that displays a static image or dynamic image stream (video).

    `QImage` (as most monitors) supports only 8-bit colors, so any higher bpc
    images are downconverted.

    Parameters
    ----------
    aspect_ratio : int or float or None, optional
        `int`, `float`:
            Fixed aspect ratio (width/height) of scaled label.
        `None`:
            Fluent aspect ratio, resizes with label space.
    *args, **kwargs
        Arguments are passed to the `QtWidgets.QWidget` constructor.

    Signals
    -------
    sUpdateImage : QtCore.pyqtSignal(QtGui.QImage)
        A new image is ready to be displayed.
    """

    sUpdateImage = pyqtSignal(QImage)

    def __init__(self, *args, aspect_ratio=1, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_logic()
        self._init_ui(aspect_ratio=aspect_ratio)
        self._init_connections()

    def _init_logic(self):
        self._cvt_func_bpc = None
        self._bytes_per_pixel = None
        self._rgb_swapped = None
        self._image_format = None

    def _init_ui(self, aspect_ratio=1):
        self.setWindowTitle("QtImage")
        self.main_layout = QHBoxLayout()
        self.setLayout(self.main_layout)
        self.image = QScaledLabel(self, aspect_ratio=aspect_ratio)
        self.image.setFrameStyle(QFrame.Panel)
        self.main_layout.addWidget(self.image)

    def _init_connections(self):
        self.sUpdateImage.connect(self.set_image)

    @staticmethod
    def _scale_bpc(val, bpc_source, bpc_target):
        """
        Proportionaly rescales the dynamic range of a value.

        Parameters
        ----------
        val : int
            Channel intensity value on the `bpc_source`-scale.
        bpc_source : int
            Source image bit range.
        bpc_target : int
            Target image bit range.

        Returns
        -------
        val_target : int
            Channel intensity value on the `bpc-target`-scale.

        Examples
        --------
        >>> bpc_src, bpc_trg = 10, 8
        >>> val = 2**bpc_src - 1  # half intensity on source scale
        >>> QtImage._scale_bpc(val, bpc_src, bpc_trg) == 2**bpc_trg - 1
        True
        """
        return int(round(
            float(2**bpc_target - 1) / float(2**bpc_source - 1) * val
        ))

    def set_image_format(self, channel="rgb", bpc=8):
        """
        Sets the image format of the incoming image stream.

        Parameters
        ----------
        channel : str or None
            str: "rgb", "rgba", "bgr", "bgra", "mono"
            None: "mono" but array without channel dimension
        bpc : int
            Bits per channel: 8, 10, 12

        See Also
        --------
        drv.itf.camcfg.CameraCfg
        """
        # Set channel and image format conversions
        if channel == "rgb":
            self._image_format = QImage.Format_RGB888
            self._bytes_per_pixel = 3
            self._rgb_swapped = False
        elif channel == "rgba":
            self._image_format = QImage.Format_RGBA8888
            self._bytes_per_pixel = 4
            self._rgb_swapped = False
        elif channel == "bgr":
            self._image_format = QImage.Format_RGB888
            self._bytes_per_pixel = 3
            self._rgb_swapped = True
        elif channel == "bgra":
            self._image_format = QImage.Format_RGBA8888
            self._bytes_per_pixel = 4
            self._rgb_swapped = True
        elif channel == "mono" or channel is None:
            self._image_format = QImage.Format_Grayscale8
            self._bytes_per_pixel = 1
            self._rgb_swapped = False
        # Set bpc conversions
        if bpc == 8:
            self._cvt_func_bpc = None
        else:
            self._cvt_func_bpc = lambda val: QtImage._scale_bpc(val, bpc, 8)

    def update_image(self, np_image):
        """
        Converts a given `numpy.ndarray` image into a `QtGui.QPixmap` image
        and emits a signal updating the image widget.

        Parameters
        ----------
        np_image : numpy.ndarray(3)
            Integer numpy array with a format as specified using the
            `set_image_format` method.

        Returns
        -------
        image : QtGui.QImage
            Image in Qt format.

        Emits
        -----
        : QtCore.pyqtSignal

        Notes
        -----
        Warning: Pass the image in the correct format as no checks are run.
        """
        if self._cvt_func_bpc is not None:
            np_image = self._cvt_func_bpc(np_image)
        height, width = np_image.shape[0], np_image.shape[1]
        bytes_per_line = width * self._bytes_per_pixel
        # create deep copy of the image to retain the data
        np_image = np_image.copy()
        image = QImage(np_image, width, height, bytes_per_line,
                       self._image_format)
        if self._rgb_swapped:
            image = image.rgbSwapped()
        self.sUpdateImage.emit(image)

    @pyqtSlot(QImage)
    def set_image(self, image):
        self.image.set_pixmap(QPixmap.fromImage(image))


###############################################################################


if __name__ == "__main__":

    # Test Imports
    import numpy as np
    import sys
    import time
    try:
        from . import addpath   # noqa
    except(ImportError):
        import addpath          # noqa
    import util.thread as thread

    # Random image generator
    def get_random_8bit_array(shape, seed=0):
        size = 1
        for s in shape:
            size *= s
        rand_state = np.random.RandomState(seed)
        arr = rand_state.randint(0, 255, size, "uint8")
        arr = arr.reshape(shape)
        return arr

    # Random image source
    class ImageSource():

        def __init__(self, qt_image, fps=50.0, duration=10.0):
            super().__init__()
            self.timer = thread.PeriodicTimer(1.0 / fps, self.update)
            self.shape = (1080, 1920, 3)
            self.image_count = int(fps * duration)
            self.images = [get_random_8bit_array(self.shape, seed=it)
                           for it in range(self.image_count)]
            self.qt_image = qt_image
            self.counter = None
            self.timer_start = None

        def update(self):
            t0 = time.time()
            self.qt_image.update_image(
                self.images[self.counter % len(self.images)]
            )
            t1 = time.time()
            print(("Update image {:d}: frame processing: {:.0f}µs, " +
                   "total: {:.1f}s").format(
                        self.counter,
                        float(t1 - t0) * 1e6,
                        float(t1 - self.timer_start)))
            self.counter += 1

        def run(self):
            print("Initial sleep for 2 seconds")
            time.sleep(2)
            self.counter = 0
            self.timer_start = time.time()
            self.timer.start()

    # Test settings
    fps = 50.0
    duration = 10.0
    aspect_ratio = 0.5

    # Create image widget
    app = QApplication(sys.argv)
    print("Setting up QtImage")
    qt_image = QtImage(aspect_ratio=aspect_ratio)
    qt_image.set_image_format(channel="rgb", bpc=8)
    qt_image.show()

    # Create RGB test image source
    print("Setting up ImageSource")
    image_source = ImageSource(qt_image, fps=fps, duration=duration)

    # Run test slide show
    print("Start video")
    image_source.run()
    sys.exit(app.exec_())
