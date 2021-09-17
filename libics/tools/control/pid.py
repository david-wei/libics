import numpy as np
from uuid import uuid4 as uuid

from libics.env import logging
from libics.tools.math import intervalfunc as ivf


###############################################################################
# 2D array control loop
###############################################################################


def get_dif_image(act_image, trg_image, offset=None, scale=None):
    """
    Calculates the difference between actual and target image.

    Parameters
    ----------
    act_image : `Array[2, float]`
        Actual image.
    trg_image : `Array[2, float]`
        Target image.
    offset, scale
        TODO: rescaling stuff

    Returns
    -------
    dif_image : `Array[2, float]`
        Difference image.
    """
    dif_image = act_image - trg_image
    return dif_image


def get_err_image(dif_images, kernel, vmin=None, vmax=None):
    """
    Calculates the error signal from a history of difference images.

    Parameters
    ----------
    dif_images : `Array[3, float]`
        History of difference images with dimensions:
        `[history_steps, *image_dims]`.
    kernel : `Array[1, float]`
        Weighing gain kernel.
        Can be used to implement PI control.
    vmin, vmax : `float`
        Clipping values.

    Returns
    -------
    err_image : `Array[2, float]`
        Error signal image.
    """
    kernel = np.array(kernel)[:, np.newaxis, np.newaxis]
    dif_images = np.array(dif_images)
    if len(kernel) < len(dif_images):
        dif_images = dif_images[-len(kernel):]
    elif len(kernel) > len(dif_images):
        kernel = kernel[-len(dif_images):]
    err_image = np.sum(dif_images * kernel, axis=0)
    if vmin:
        err_image[err_image < vmin] = vmin
    if vmax:
        err_image[err_image > vmax] = vmax
    return err_image


def get_ctrl_image(old_ctrl_image, err_image):
    ctrl_image = old_ctrl_image - err_image
    return ctrl_image


class ImageControlLoop(object):

    LOGGER = logging.get_logger("libics.tools.control.pid.ImageControlLoop")

    def __init__(self, *args, trg_image=None, init_ctrl_image=None, **kwargs):
        # ID
        self.id = str(uuid())
        self.folder = ""
        # Images
        self.trg_image = trg_image
        # History
        self.act_images = []
        self.dif_images = []
        self.err_images = []
        self.ctrl_kernels = []
        self.ctrl_images = [init_ctrl_image]
        # Control loop
        self.ctrl_kernel = None
        self.ctrl_gain_integr_unlim = None
        # Parse args
        if len(args) != 0:
            raise ValueError
        # Parse kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __len__(self):
        return len(self.ctrl_images)

    def set_ctrl_kernel(
        self, gain_prop, gain_integr_lim, num_integr_lim, tau_integr_lim=-1e6
    ):
        """
        Sets feedback parameters for proportional and limited integral gain.

        For the limited integral control loop, an exponential kernel is used.

        Parameters
        ----------
        amp_prop : `float`
            Proportional gain.
        gain_integr_lim : `float`
            Limited integral gain at latest step.
            Exponentially taperes off over characteristic length
            number of steps towards zero.
        num_integr_lim : `int`
            Number of steps until limited integral gain is zero.
        tau_integr_lim : `float`
            Exponential characteristic number of steps,
            i.e. denominator of exponent.
        """
        _x = np.linspace(0, 1, num=num_integr_lim)
        kernel = np.zeros(num_integr_lim + 1, dtype=float)
        kernel[:-1] = ivf.exp(_x, gain_integr_lim, 0, tau_integr_lim)
        kernel[-1] += gain_prop
        self.ctrl_kernel = kernel

    def get_ctrl_kernel(self):
        """Gets the PI-kernel used to calculate the error signal."""
        kernel = self.ctrl_kernel
        if kernel is None:
            kernel = np.zeros(1, dtype=float)
        if self.ctrl_gain_integr_unlim is not None:
            kernel_unlim = np.full(
                max(len(kernel), len(self)),
                self.ctrl_gain_integr_unlim, dtype=float
            )
            if len(kernel_unlim) > len(kernel):
                _kernel = np.zeros_like(kernel_unlim)
                _kernel[-len(kernel):] = kernel
                kernel = _kernel
            kernel = kernel + kernel_unlim
        return kernel

    def add_ctrl_step(self, act_image, step=None):
        """
        Adds a measurement step.

        Parameters
        ----------
        act_image : `Array[2, float]`
            Actual, measured image.
        step : `int` or `None`
            If `int`, removes all later steps and updates the given step.
            If `None`, appends a new step.
        """
        if len(self) == 0:
            raise RuntimeError("no initial control image")
        if step is not None:
            self.act_images = self.act_images[:step]
            self.dif_images = self.dif_images[:step]
            self.err_images = self.err_images[:step]
            self.ctrl_kernels = self.ctrl_kernels[:step]
            self.ctrl_images = self.ctrl_images[:step+1]
        self.act_images.append(act_image)
        self.dif_images.append(get_dif_image(act_image, self.trg_image))
        self.ctrl_kernels.append(self.get_ctrl_kernel())
        self.err_images.append(get_err_image(
            self.dif_images, self.ctrl_kernels[-1]
        ))
        self.ctrl_images.append(get_ctrl_image(
            self.ctrl_images[-1], self.err_images[-1],
            vmin=self.ctrl_min, vmax=self.ctrl_max
        ))
