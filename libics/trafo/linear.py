import numpy as np
from scipy import ndimage, optimize

from libics.file import hdf
from libics.util import InheritMap


###############################################################################


@InheritMap(map_key=("libics", "AffineTrafo"))
class AffineTrafo(hdf.HDFBase):

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

    def __init__(
        self,
        matrix=np.diag([1, 1]).astype(float),
        offset=np.array([0, 0], dtype=float)
    ):
        super().__init__(pkg_name="libics", cls_name="AffineTrafo")
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

        Parameter
        ---------
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

        Parameter
        ---------
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
