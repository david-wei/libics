import numpy as np
from PyQt5.QtGui import QImage


def convert_qimage_to_numpy_ndarray(qt_image, im_format=None, channel="mono"):
    """
    Converts a QtGui.QImage into an numpy.ndarray.

    Parameters
    ----------
    qt_image : QtGui.QImage
        The image in Qt format.
    im_format : QImage.Format or None
        Sets `QImage` image format.
        If `None`, uses `channel` parameter, otherwise
        overwrites it.
    channel : str or None
        Numpy image format: `"rgba"`, `"rgb"`, `"mono"`, `None`.

    Returns
    -------
    np_image : numpy.ndarray
        The image in numpy format.
    """
    np_image = None
    bytes_per_px = None
    defined_format = (im_format is not None)
    if im_format is None:
        if channel is None or channel == "mono":
            im_format = QImage.Format_Grayscale8
        elif channel == "rgb":
            im_format = QImage.Format_RGB888
        elif channel == "rgba":
            im_format = QImage.Format_RGBA8888
    if im_format == QImage.Format_Grayscale8:
        if defined_format or channel is not None:
            bytes_per_px = 1
        else:
            bytes_per_px = None
    elif im_format == QImage.Format_RGB888:
        bytes_per_px = 3
    elif im_format == QImage.Format_RGBA8888:
        bytes_per_px = 4
    qt_image = qt_image.convertToFormat(im_format)
    width, height = qt_image.width(), qt_image.height()
    ptr = qt_image.bits()
    ptr.setsize(qt_image.byteCount())
    if bytes_per_px is None:
        np_image = np.array(ptr).reshape(height, width)
    else:
        np_image = np.array(ptr).reshape(height, width, bytes_per_px)
    return np_image
