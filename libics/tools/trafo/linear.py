import numpy as np
from scipy import ndimage, optimize

from libics.env import logging
from libics.core.io import FileBase
from libics.tools.math.fit import split_fit_data
from libics.tools.math.peaked import FitGaussian2dTilt


###############################################################################


class AffineTrafo(FileBase):

    """
    Defines an affine transformation in arbitrary dimensions.

    Parameters
    ----------
    matrix : np.ndarray(2, float)
        Transformation matrix.
    offset : np.ndarray(1, float)
        Transformation offset.

    # TODO: transformation of array data.
    """

    SER_KEYS = FileBase.SER_KEYS | {"matrix", "offset"}

    def __init__(
        self,
        matrix=np.diag([1, 1]).astype(float),
        offset=np.array([0, 0], dtype=float)
    ):
        self.matrix = matrix
        self.offset = offset

    def fit_affine_transform(self, origin_coords, target_coords):
        """
        Fits the affine transform matrix and offset vector.

        Parameters
        ----------
        origin_coords, target_coords : list(np.ndarray(1, float))
            List of (origin, target) coordinates in corresponding order.

        Returns
        -------
        success : bool
            Whether transformation fit succeeded.
            Matrix and offset attributes are only written
            in the case of success.

        Notes
        -----
        Following H. Späth, Math. Com. 9 (1), 27-34.
        Variable naming convention:
        * q: origin coordinates.
        * p: target coordinates.
        * m: transform matrix.
        * b: transform offset.
        """
        # Variables
        q = np.array([
            np.concatenate((np.array(origin), [1.0]))
            for origin in origin_coords
        ], dtype=float)
        p = np.array(target_coords, dtype=float)
        # Derived variables
        dim = q[0].shape[0] - 1
        Q = np.sum([np.outer(qq, qq) for qq in q], axis=0)
        a = np.full((dim, dim + 1), np.nan, dtype=float)
        c = np.array([np.dot(q.T, pp) for pp in p.T])
        # Optimization
        for i, cc in enumerate(c):
            res = optimize.lsq_linear(Q, cc)
            if not res.success:
                return False
            a[i] = res.x
        # Assignment
        m = a[:dim, :dim]
        b = a.T[-1]
        self.matrix = m
        self.offset = b
        return True

    # ++++ Perform transformation ++++

    def __call__(self, image, shape, order=3, direction="to_target"):
        if direction == "to_target":
            return self.cv_to_target(image, shape, order=order)
        elif direction == "to_origin":
            return self.cv_to_origin(image, shape, order=order)

    def cv_to_origin(self, image, shape, order=3):
        """
        Convert an image in target coordinates to origin coordinates.

        Parameters
        ----------
        image : np.ndarray(2, float)
            Image in target coordinates.
        shape : tuple(int)
            Shape of trafo_image.
        order : int
            Spline interpolation order.

        Returns
        -------
        trafo_image : np.ndarray(2, float)
            Image in origin coordinates.
        """
        return ndimage.affine_transform(
            image, self.matrix, offset=self.offset, output_shape=shape,
            order=order, mode="constant", cval=0.0
        )

    def cv_to_target(self, image, shape, order=3):
        """
        Convert an image in origin coordinates to target coordinates.

        Parameters
        ----------
        image : np.ndarray(2, float)
            Image in origin coordinates.
        shape : tuple(int)
            Shape of trafo_image.
        order : int
            Spline interpolation order.

        Returns
        -------
        trafo_image : np.ndarray(2, float)
            Image in target coordinates.
        """
        inv_matrix = np.linalg.inv(self.matrix)
        inv_offset = -np.dot(inv_matrix, self.offset)
        return ndimage.affine_transform(
            image, inv_matrix, offset=inv_offset, output_shape=shape,
            order=order, mode="constant", cval=0.0
        )


class AffineTrafo2d(AffineTrafo):

    """
    Maps origin pixel positions to target pixels positions.

    Convention: ``target_coord = matrix * origin_coord + offset``.

    Usage:

    * Take images for different single-pixel illuminations.
    * Calculate transformation parameters with :py:meth:`calc_trafo`.
    * Perform transforms with call method
      (or :py:meth:`cv_to_target`, :py:meth:`cv_to_origin`).
    """

    LOGGER = logging.get_logger("libics.tools.trafo.linear.AffineTrafo2d")

    def __init__(
        self,
        matrix=np.diag([1, 1]).astype(float),
        offset=np.array([0, 0], dtype=float)
    ):
        self._matrix = None
        self._offset = None
        self._matrix_scale = None
        self._matrix_rotation = None
        self._matrix_shear = None
        super().__init__(matrix=matrix, offset=offset)

    @property
    def matrix(self):
        return self._matrix

    @matrix.setter
    def matrix(self, val):
        self._matrix = val
        mat_dec = self._decompose_matrix(val)
        self._matrix_scale, self._matrix_shear, self._matrix_rotation = mat_dec

    @property
    def origin_scale(self):
        pass

    @property
    def target_scale(self):
        pass

    @property
    def matrix_scale(self):
        return self._matrix_scale

    @property
    def origin_shear(self):
        pass

    @property
    def target_shear(self):
        pass

    @property
    def matrix_shear(self):
        return self._matrix_shear

    @property
    def origin_rotation(self):
        pass

    @property
    def target_rotation(self):
        pass

    @property
    def matrix_rotation(self):
        return self._matrix_rotation

    @property
    def origin_offset(self):
        pass

    @property
    def target_offset(self):
        pass

    @staticmethod
    def _decompose_matrix(mat):
        """
        Decomposes an affine transformation matrix.

        Parameters
        ----------
        mat : `np.ndarray(2, float)`
            Affine transformation matrix.

        Returns
        -------
        mat_scale, mat_shear, mat_rotation : `np.ndarray(2, float)`
            Scaling, shear, rotation matrices.
            Matrices are applied in above order,
            i.e. ``M_rot M_shear M_scale vec``.
        """
        raise NotImplementedError

    def fit_peak_coordinates(self, image, snr=1.5):
        """
        Uses a Gaussian fit to obtain the peak coordinates of an image.

        Parameters
        ----------
        image : `np.ndarray(2, float)` or `ArrayData`
            Image to be analyzed.
        snr : `float`
            Maximum-to-mean ratio required for fit.

        Returns
        -------
        x, y : float, None
            (Fractional) image index coordinates of fit position.
            None: If fit failed.
        """
        max_, mean, sum_ = image.max(), image.mean(), image.sum()
        if (max_ == 0 or mean == 0 or max_ / mean < snr
                or max_ / sum_ < 1 / 100**2):
            self.LOGGER.warning("fit_peak_coordinates: insufficient SNR")
            return None
        try:
            self.LOGGER.debug("fit_peak_coordinates: fitting curve")
            _fit = FitGaussian2dTilt()
            _data = split_fit_data(image)
            _fit.find_init_param(*_data)
            _fit.find_fit(*_data)
            x, y = _fit.param[0], _fit.param[1]
            dx, dy = _fit.std[0], _fit.std[1]
        except (TypeError, RuntimeError) as e:
            self.LOGGER.warning(
                "fit_peak_coordinates: fit failed ({:s})".format(str(e))
            )
            return None
        x, y = abs(x), abs(y)
        if (x != 0 and dx / x > 0.2) or (y != 0 and dy / y > 0.2):
            self.LOGGER.warning("fit_peak_coordinates: did not converge")
            return None
        return np.array([x, y])

    def calc_trafo(self, origin, target, **kwargs):
        """
        Estimates the transformation parameters.

        If images are passed, a 2D Gaussian fit is applied.

        Parameters
        ----------
        origin : `list(np.ndarray(2, float) or (x, y))`
            List of origin images or coordinates.
        target : `list(np.ndarray(2, float) or (x, y))`
            List of target images or coordinates.
        **kwargs
            Keyword arguments passed to :py:meth:`fit_peak_coordinates`.

        Returns
        -------
        ret : `bool`
            Whether calculation was successful.

        Notes
        -----
        * Coordinate and image order must be identical.
        * At least two images are required. Additional data is used to obtain
          a more accurate map.
        """
        if min(len(origin), len(target)) <= 1:
            raise ValueError("not enough images or coordinates")
        if len(origin) != len(target):
            self.LOGGER.warning("unequal origin/target list length")
        origin_coords = []
        target_coords = []
        for i in range(min(len(origin), len(target))):
            _origin = np.array(origin[i], dtype=float)
            _target = np.array(target[i], dtype=float)
            if _origin.ndim != 1:
                _origin = self.fit_peak_coordinates(_origin, **kwargs)
                if _origin is None:
                    continue
            if _target.ndim != 1:
                _target = self.fit_peak_coordinates(_target, **kwargs)
                if _target is None:
                    continue
            origin_coords.append(_origin)
            target_coords.append(_target)
        origin_coords = np.array(origin_coords)
        target_coords = np.array(target_coords)
        if len(origin_coords) >= 2:
            self.fit_affine_transform(origin_coords, target_coords)
            return True
        else:
            self.LOGGER.warning("not enough coordinates extracted from images")
            return False


###############################################################################


if __name__ == "__main__":

    # Affine transformation test data
    target_coords = np.array([
        (0, 0), (767, 0), (0, 767), (767, 767)
    ])
    origin_coords = np.array([
        (1037, 519), (519, 1037), (519, 0), (0, 519)
    ])
    im_target = np.zeros((1024, 768))
    for i in range(1024):
        for j in range(768):
            if i < 768 and j < 768:
                if i < 250:
                    im_target[i, j] = 0.5
                else:
                    im_target[i, j] = 1
                if i < j:
                    im_target[i, j] *= 0.7
    im_origin = np.zeros((1388, 1038))
    for i in range(1388):
        for j in range(1038):
            if i < 1038 and j < 1038:
                if i < 300:
                    im_origin[i, j] = 0.5
                else:
                    im_origin[i, j] = 1
                if i < j:
                    im_origin[i, j] *= 0.7

    # Perform affine transform
    at = AffineTrafo()
    if not at.fit_affine_transform(origin_coords, target_coords):
        print("at failed")
    im_origin_at = at(im_origin, (1024, 768))
    im_target_at = at(im_target, (1388, 1038), direction="to_origin")

    # Plot test
    import matplotlib.pyplot as plt

    def plot(im, subp, title):
        plt.subplot(subp)
        plt.pcolormesh(im.T, vmin=0, vmax=1)
        plt.gca().set_aspect(1)
        plt.title(title)
        plt.colorbar()
    plot(im_origin, 221, "origin")
    plot(im_origin_at, 222, "AT-> target")
    plot(im_target, 223, "target")
    plot(im_target_at, 224, "AT-> origin")
    plt.tight_layout()
    plt.show()
