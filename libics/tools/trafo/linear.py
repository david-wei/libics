import numpy as np
from scipy import ndimage, optimize
import time

from libics.env import logging
from libics.core.data.arrays import ArrayData
from libics.core.io import FileBase
from libics.core.util import misc
from libics.tools.math.peaked import FitGaussian2dTilt


###############################################################################


class AffineTrafo(FileBase):

    """
    Defines an affine transformation in arbitrary dimensions.

    Parameters
    ----------
    matrix : `np.ndarray(2, float)`
        Transformation matrix.
    offset : `np.ndarray(1, float)`
        Transformation offset.
    """

    LOGGER = logging.get_logger("libics.tools.trafo.linear.AffineTrafo")
    SER_KEYS = FileBase.SER_KEYS | {"matrix", "offset"}

    def __init__(
        self,
        matrix=np.diag([1, 1]).astype(float),
        offset=np.array([0, 0], dtype=float)
    ):
        self.matrix = matrix
        self.offset = offset

    @property
    def matrix_to_target(self):
        return self.matrix

    @property
    def matrix_to_origin(self):
        return np.linalg.inv(self.matrix)

    @property
    def offset_to_target(self):
        return self.offset

    @property
    def offset_to_origin(self):
        return -np.dot(self.matrix_to_origin, self.offset)

    def fit_affine_transform(self, origin_coords, target_coords):
        """
        Fits the affine transform matrix and offset vector.

        Parameters
        ----------
        origin_coords, target_coords : `list(np.ndarray(1, float))`
            List of (origin, target) coordinates in corresponding order.

        Returns
        -------
        success : `bool`
            Whether transformation fit succeeded.
            Matrix and offset attributes are only written
            in the case of success.

        Notes
        -----
        Following H. Späth, Math. Com. 9 (1), 27-34 (2004).
        Variable naming convention:

        * `q`: origin coordinates.
        * `p`: target coordinates.
        * `m`: transform matrix.
        * `b`: transform offset.
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

    def __call__(self, *args, direction="to_target", **kwargs):
        """
        Performs a transformation.

        Parameters
        ----------
        direction : `string`
            `"to_target"`: see :py:meth:`cv_to_target`.
            `"to_origin"`: see :py:meth:`cv_to_origin`.
        """
        if direction == "to_target":
            return self.cv_to_target(*args, **kwargs)
        elif direction == "to_origin":
            return self.cv_to_origin(*args, **kwargs)

    def coord_to_origin(self, target_coords):
        """
        Transforms given target coordinates into origin coordinates.

        See :py:meth:`coord_to_target`.
        """
        ct = np.array(target_coords)
        mot = self.matrix_to_origin
        bot = self.offset_to_origin
        co = np.einsum("ij,...j->...i", mot, ct) + bot
        return co

    def coord_to_target(self, origin_coords):
        """
        Transforms given origin coordinates into target coordinates.

        Parameters
        ----------
        origin_coords : `np.ndarray(float)`
            Coordinates in origin space. The different dimensions should be
            placed on the last axes (dimensions: [..., ndim]).

        Returns
        -------
        target_coords : `np.ndarray(float)`
            Transformed coordinates in target space.
        """
        co = np.array(origin_coords)
        mto = self.matrix_to_target
        bto = self.offset_to_target
        ct = np.einsum("ij,...j->...i", mto, co) + bto
        return ct

    def cv_to_origin(
        self, target_array, origin_shape, centered=False, supersample=None,
        **kwargs
    ):
        """
        Convert an array in target coordinates to origin coordinates.

        See :py:meth:`cv_to_target`.
        """
        matrix = np.linalg.inv(self.matrix)
        offset = -np.dot(matrix, self.offset)
        return self._cv_array(
            target_array, origin_shape, matrix, offset,
            centered=centered, supersample=supersample, **kwargs
        )

    def cv_to_target(
        self, origin_array, target_shape, centered=False, supersample=None,
        **kwargs
    ):
        """
        Convert an array in origin coordinates to target coordinates.

        Parameters
        ----------
        origin_array : `np.ndarray(float)` or `ArrayData(float)`
            Array in origin coordinates.
        target_shape : `tuple(int)` or `ArrayData(float)`
            Shape of target array.
            If `ArrayData`, loads the transformed array into this object.
        centered : `bool`
            Only relevant if `target_shape` is `tuple(int)`.
            If `True`, sets the origin zero to the center of the array.
            Otherwise assumes the center to be at index `(0, 0)`.
        supersample : `int`
            Origin array supersampling repetitions (for sharper edges).
        **kwargs
            See scipy documentation for ``scipy.ndimage.map_coordinates``.
            Notable options: `order : int` (interpolation order).

        Returns
        -------
        target_array : `np.ndarray(float)` or `ArrayData(float)`
            Array transformed to target coordinates.
            Return type depends on `target_shape` type.

        Notes
        -----
        * Performs subsequent transformations from image to origin to target
          coordinates.
        * Interpolates into the requested target shape.
        * For algorithm see :py:meth:`_cv_array`.
        """
        return self._cv_array(
            origin_array, target_shape, self.matrix, self.offset,
            centered=centered, supersample=supersample, **kwargs
        )

    @staticmethod
    def _cv_array(
        from_array, to_shape, matrix, offset, centered=False, supersample=None,
        **kwargs
    ):
        r"""
        Performs the affine transformation.

        For parameters see :py:meth:`cv_to_target` as example.

        Notes
        -----
        Variables: coordinate :math:`\mathbf{c}`, to-system :math:`t`,
        from-system :math:`f`, image-system :math:`i`,
        transformation matrix :math:`\mathrm{M}`,
        transformation offset :math:`\mathbf{b}`.

        .. math::
            \mathbf{c}_t = \mathrm{M}_{ti}\mathbf{c}_i+\mathbf{b}_{ti},\\
            \mathrm{M}_{ti} = \mathrm{M}_{tf}\mathrm{M}_{fi},\\
            \mathbf{b}_{ti} = \mathrm{M}_{tf}\mathbf{b}_{fi}+\mathbf{b}_{tf},\\
            \mathbf{c}_i = \mathrm{M}_{ti}^{-1}\mathbf{c}_t
                           - \mathrm{M}_{ti}^{-1}\mathbf{b}_{ti}.
        """
        # Extract from_array parameters
        if not isinstance(from_array, ArrayData):
            _from_array = ArrayData()
            if centered is True:
                for i in range(from_array.ndim):
                    _from_array.add_dim(
                        offset=(from_array.shape[i] - 1) / 2, step=1
                    )
            else:
                _from_array.add_dim(from_array.ndim)
            _from_array.data = from_array
            from_array = _from_array

        _step = np.array(from_array.step)
        _low = np.array(from_array.low)
        _from = from_array.data

        if supersample is not None:
            AffineTrafo.LOGGER.error("TODO: Supersampling seems to have bugs!")
            supersample = round(supersample)
            _step = _step / supersample
            _low = _low - _step / supersample / 2
            _from = AffineTrafo._supersample(_from, supersample)
        # Extract to_shape parameters
        if isinstance(to_shape, ArrayData):
            _coord_t = to_shape.get_var_meshgrid()
        else:
            if centered is True:
                _coord_t = np.array(np.meshgrid(*[
                    np.arange(s) - s // 2 for s in to_shape
                ], indexing="ij"), dtype=float)
            else:
                _coord_t = np.indices(to_shape, dtype=float)
        # Linear transform origin to target
        mtf = matrix
        btf = offset
        # Linear transform image to origin
        mfi = np.diag(_step)
        bfi = _low
        # Linear transform image to target
        mti = np.dot(mtf, mfi)
        bti = np.dot(mtf, bfi) + btf
        # Coordinate transformation
        ct = _coord_t
        ci = np.einsum(
            "ij,j...->i...",
            np.linalg.inv(mti),
            ct - AffineTrafo._append_dims(bti, ct.ndim - 1)
        )
        _to = ndimage.map_coordinates(_from, ci, **kwargs)
        # Return transformed array
        if isinstance(to_shape, ArrayData):
            to_shape.data = _to
            return to_shape
        else:
            return _to

    def cv_offset_shift_to_target(self, shift):
        """Converts an offset shift from origin to target space."""
        return self._cv_offset_shift(shift, self.matrix_to_target)

    def cv_offset_shift_to_origin(self, shift):
        """Converts an offset shift from target to origin space."""
        return self._cv_offset_shift(shift, self.matrix_to_origin)

    @staticmethod
    def _cv_offset_shift(shift, matrix):
        """
        Converts an offset shift between origin and target space.

        Parameters
        ----------
        shift : `Array[1, float]`
            Offset shift in initial space.
        matrix : `Array[2, float]`
            Transformation matrix.

        Returns
        -------
        shift_cv : `Array[1, float]`
            Offset shift in transformed space.
        """
        ar_shift = np.array(shift, dtype=float)
        ar_matrix = np.array(matrix, dtype=float)
        ar_shift_cv = ar_matrix @ ar_shift
        if isinstance(shift, ArrayData):
            shift_cv = shift.copy_var()
            shift_cv.data = ar_shift_cv
        else:
            shift_cv = ar_shift_cv
        return shift_cv

    # ++++ Operations on trafo ++++

    def invert(self):
        """
        Returns the inverse `AffineTrafo` object.
        """
        return self.__class__(
            matrix=self.matrix_to_origin, offset=self.offset_to_origin
        )

    def __invert__(self):
        return self.invert()

    def concatenate(self, other):
        """
        Returns an `AffineTrafo` object which concatenates two transformations.

        First transform: `other`, second transform: `self`.
        """
        matrix = self.matrix @ other.matrix
        offset = self.matrix @ other.offset + self.offset
        return self.__class__(matrix=matrix, offset=offset)

    def __matmul__(self, other):
        return self.concatenate(other)

    # ++++ Helper functions ++++

    @staticmethod
    def _append_dims(ar, dims=1):
        """Appends `dims` empty dimensions to the given numpy `ar`ray."""
        return ar.reshape((-1,) + (1,) * dims)

    @staticmethod
    def _supersample(ar, rep):
        """Supersamples an `ar`ray in all dimensions by `rep`etitions."""
        ar = np.array(ar)
        if not np.isscalar(rep):
            rep = int(round(rep))
        if rep == 1:
            return ar
        elif rep < 1:
            raise ValueError(f"invalid repetitions {str(rep):s}")
        for i in range(ar.ndim):
            ar = ar.repeat(rep, axis=i)
        return ar


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
        super().__init__(matrix=matrix, offset=offset)

    def set_target_axes(
        self,
        magnification=np.ones(2, dtype=float),
        angle=np.zeros(2, dtype=float),
        offset=np.zeros(2, dtype=float)
    ):
        """
        Sets the target coordinate system axes.

        Parameters
        ----------
        magnification : `(float, float)`
            Length of unit vectors in origin units.
        angle : `(float, float)`
            Angle of unit vectors with respect to origin axes in radians (rad).
        offset : `(float, float)`
            Coordinate system zero position in origin units.
        """
        # Coordinates: (u, v) target, (x, y) origin
        # Transformation: c_t = M_to c_o + b_to
        mu, mv = magnification
        thu, thv = angle
        bot = offset
        mot = np.array([
            [mu * np.cos(thu), -mv * np.sin(thv)],
            [mu * np.sin(thu), mv * np.cos(thv)]
        ])
        self.matrix = np.linalg.inv(mot)
        self.offset = -np.dot(self.matrix, bot)

    def get_target_axes(self):
        """
        Returns
        -------
        magnification : `(float, float)`
            Length of unit vectors in origin units.
        angle : `(float, float)`
            Angle of unit vectors with respect to origin axes in radians (rad).
        offset : `(float, float)`
            Coordinate system zero position in origin units.
        """
        mot = self.matrix_to_origin
        bot = self.offset_to_origin
        thu = np.arctan2(mot[1, 0], mot[0, 0])
        thv = np.arctan2(-mot[0, 1], mot[1, 1])
        mu = np.sqrt(mot[0, 0]**2 + mot[1, 0]**2)
        mv = np.sqrt(mot[0, 1]**2 + mot[1, 1]**2)
        return (
            np.array([mu, mv]),
            np.array([thu, thv]),
            np.array(bot)
        )

    def set_origin_axes(
        self,
        magnification=np.ones(2, dtype=float),
        angle=np.zeros(2, dtype=float),
        offset=np.zeros(2, dtype=float)
    ):
        """
        Sets the origin coordinate system axes.

        Parameters
        ----------
        magnification : `(float, float)`
            Length of unit vectors in target units.
        angle : `(float, float)`
            Angle of unit vectors with respect to target axes in radians (rad).
        offset : `(float, float)`
            Coordinate system zero position in target units.
        """
        # Coordinates: (u, v) target, (x, y) origin
        # Transformation: c_t = M_to c_o + b_to
        mx, my = magnification
        thx, thy = angle
        bto = offset
        mto = np.array([
            [mx * np.cos(thx), -my * np.sin(thy)],
            [mx * np.sin(thx), my * np.cos(thy)]
        ])
        self.matrix = mto
        self.offset = bto

    def get_origin_axes(self):
        """
        Returns
        -------
        magnification : `(float, float)`
            Length of unit vectors in target units.
        angle : `(float, float)`
            Angle of unit vectors with respect to target axes in radians (rad).
        offset : `(float, float)`
            Coordinate system zero position in target units.
        """
        mto = self.matrix_to_target
        bto = self.offset_to_target
        thx = np.arctan2(mto[1, 0], mto[0, 0])
        thy = np.arctan2(-mto[0, 1], mto[1, 1])
        mx = np.sqrt(mto[0, 0]**2 + mto[1, 0]**2)
        my = np.sqrt(mto[0, 1]**2 + mto[1, 1]**2)
        return (
            np.array([mx, my]),
            np.array([thx, thy]),
            np.array(bto)
        )

    # +++++++++++++++++++++++++++++++++++++++++

    def __get_str_origin_axes(self):
        m, th, b = self.get_origin_axes()
        return f"M = {str(m)}, θ = {str(np.rad2deg(th))}, b = {str(b)}"

    def __str__(self):
        return f"AffineTrafo2d: origin_axes: {self.__get_str_origin_axes()}"

    def __repr__(self):
        s = f"<'{self.__class__.__name__}' at {hex(id(self))}>\n"
        s += f"{self.__get_str_origin_axes()}"
        return s

    # +++++++++++++++++++++++++++++++++++++++++

    def fit_peak_coordinates(self, image, snr=1.5, max_sum_ratio=1/100**2):
        """
        Uses a Gaussian fit to obtain the peak coordinates of an image.

        Parameters
        ----------
        image : `np.ndarray(2, float)` or `ArrayData`
            Image to be analyzed.
        snr : `float`
            Maximum-to-mean ratio required for fit.
        max_sum_ratio : `float`
            Maximum-to-sum ratio required for fit.

        Returns
        -------
        x, y : float, None
            (Fractional) image index coordinates of fit position.
            None: If fit failed.
        """
        max_, mean, sum_ = image.max(), image.mean(), image.sum()
        if (max_ == 0 or mean == 0 or max_ / mean < snr
                or max_ / sum_ < max_sum_ratio):
            self.LOGGER.warning("fit_peak_coordinates: insufficient SNR")
            return None
        try:
            self.LOGGER.debug("fit_peak_coordinates: fitting curve")
            _fit = FitGaussian2dTilt()
            _fit.find_p0(image)
            _fit.find_popt(image)
            x, y = _fit.x0, _fit.y0
            dx, dy = _fit.x0_std, _fit.y0_std
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

    def find_peak_coordinates(
        self, origin, target, algorithm="fit", print_progress=False, **kwargs
    ):
        """
        Finds peak coordinates.

        See :py:meth:`calc_trafo`.
        """
        # Find coordinates
        origin_coords = []
        target_coords = []
        list_len = min(len(origin), len(target))
        t0 = time.time()
        for i in range(list_len):
            if print_progress:
                misc.print_progress(i, list_len, start_time=t0)
            _origin = np.array(origin[i], dtype=float)
            _target = np.array(target[i], dtype=float)
            if _origin.ndim != 1:
                if algorithm == "fit":
                    _origin = self.fit_peak_coordinates(_origin, **kwargs)
                else:
                    _origin = np.unravel_index(_origin.argmax(), _origin.shape)
                if _origin is None:
                    continue
            if _target.ndim != 1:
                if algorithm == "fit":
                    _target = self.fit_peak_coordinates(_target, **kwargs)
                elif algorithm == "max":
                    _target = np.unravel_index(_target.argmax(), _target.shape)
                if _target is None:
                    continue
            origin_coords.append(_origin)
            target_coords.append(_target)
        if print_progress:
            misc.print_progress(list_len, list_len, start_time=t0)
        return np.array(origin_coords), np.array(target_coords)

    def calc_trafo(
        self, origin, target, algorithm="fit", print_progress=False, **kwargs
    ):
        """
        Estimates the transformation parameters.

        Wrapper for peak coordinate finding and transformation fitting.

        Parameters
        ----------
        origin : `list(np.ndarray(2, float) or (x, y))`
            List of origin images or coordinates.
        target : `list(np.ndarray(2, float) or (x, y))`
            List of target images or coordinates.
        algorithm : `str`
            Algorithm to use to determine coordinates for given image.
            `"fit", "max"`.
        print_progress : `bool`
            Whether to print a progress bar to console.
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
        # Check parameters
        if min(len(origin), len(target)) <= 1:
            raise ValueError("not enough images or coordinates")
        if len(origin) != len(target):
            self.LOGGER.warning("unequal origin/target list length")
        if algorithm not in ["fit", "max"]:
            self.LOGGER.warning(f"invalid algorithm ({algorithm}), using max")
            algorithm = "max"
        # Fit transformation
        origin_coords, target_coords = self.find_peak_coordinates(
            origin, target,
            algorithm=algorithm, print_progress=print_progress,
            **kwargs
        )
        if len(origin_coords) >= 2:
            self.fit_affine_transform(origin_coords, target_coords)
            return True
        else:
            self.LOGGER.warning("not enough coordinates extracted from images")
            return False

    # ++++ Operations on trafo ++++

    def magnify(self, factor):
        """
        Returns an `AffineTrafo` object whose magnification is scaled.
        """
        _mag, _ang, _off = self.get_origin_axes()
        trafo = self.__class__()
        trafo.set_origin_axes(
            magnification=_mag*factor, angle=_ang, offset=_off
        )
        return trafo

    def __mul__(self, other):
        return self.magnify(other)

    def __rmul__(self, other):
        return self.magnify(other)

    def shift(self, offset):
        """
        Returns an `AffineTrafo` object whose offset is shifted.
        """
        _mag, _ang, _off = self.get_origin_axes()
        trafo = self.__class__()
        trafo.set_origin_axes(
            magnification=_mag, angle=_ang, offset=_off+offset
        )
        return trafo

    def __add__(self, other):
        return self.shift(other)

    def __radd__(self, other):
        return self.shift(other)

    def rotate(self, angle):
        """
        Returns an `AffineTrafo` object whose angle is shifted.
        """
        _mag, _ang, _off = self.get_origin_axes()
        trafo = self.__class__()
        trafo.set_origin_axes(
            magnification=_mag, angle=_ang+angle, offset=_off
        )
        return trafo


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
