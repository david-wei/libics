import copy
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
import numpy as np
import PIL
import scipy

import IPython
from IPython.display import HTML, Markdown, Latex


###############################################################################


def display_array(ar, d, cmap="viridis", vmin=None, vmax=None, height=512):
    """
    Performant function to show an image in a Jupyter cell.

    Can be used e.g. for a live view of an image sequence.

    Parameters
    ----------
    ar : `np.ndarray(2)`
        2D array to be displayed.
    d : `dict`
        Pass a new empty dictionary, i.e. `dict()`.
        This is required for the algorithm.
    cmap : `str` or `matplotlib.cm.cmaps_listed`
        Matplotlib color map.
    vmin, vmax : `int`
        Color map minimum, maximum values.
    height : `int`
        Height of image in pixels (px) to be displayed.
    """
    if "display_handle" not in d:
        d["display_handle"] = None

    if vmin is None:
        vmin = np.min(ar)
    if vmax is None:
        vmax = np.max(ar)
    ar_norm = (ar - vmin) / (vmax - vmin)
    im = getattr(mpl.cm, cmap)(ar_norm)
    im = (im * 255).astype(np.uint8)
    if im.shape[-1] == 4:
        im = im[:, :, :3]

    from io import BytesIO
    f = BytesIO()
    PIL.Image.fromarray(im).save(f, "jpeg")
    ratio = im.shape[1] / im.shape[0]
    h = height
    w = int(512 * ratio)
    display_obj = IPython.display.Image(data=f.getvalue(), width=w, height=h)
    if d["display_handle"] is None:
        d["display_handle"] = IPython.display.display(
            display_obj, display_id=True
        )
    else:
        d["display_handle"].update(display_obj)
    IPython.display.clear_output(wait=True)
