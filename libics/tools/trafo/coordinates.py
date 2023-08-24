import numpy as np

from libics.core.util import misc


###############################################################################
# Coordinate Systems
###############################################################################


def get_coords_type(*coords, **components):
    return {0: "names", 1: "coords"}.get(len(coords), "components")


def assume_coords_type(
    *coords, coord_type=None, names=None, **components
):
    """
    Converts coordinates into a specified format.

    Multidimensional arrays of multidimensional coordinates
    can be specified in three different formats:

    * `"coords"`: A single array whose coordinates are located
      in the last axis, i.e., `array.shape = (..., n_dim)`.
    * `"components"`: A list of arrays, where each list item represents
      one coordinate, i.e., `len(array) = n_dim`.
    * `"names"`: Keyword arguments, where the keyword names specify
      the respective coordinate, i.e., `len(kwargs) = n_dim`.

    Parameters
    ----------
    *coords, **components : `np.ndarray`
        Coordinates (may be given in any of the above formats).
    coord_type : `str` or `None`
        Format of the coordinates: `"coords", "components", "names"`.
        If `None`, uses the input format.
    names : `list(tuple(str))`
        Component names in the coordinate order.
        Each tuple contains aliases for the respective component.
        If `coord_type == "names"`, the first item of the tuple is used as key.

    Returns
    -------
    coords
        Coordinates in the format specified by `coord_type`.

    Raises
    ------
    ValueError
        If `coord_type` is invalid.
    IndexError, KeyError
        If `names` is incompatible with `**components`.
    """
    # Parse parameters
    input_type = get_coords_type(*coords, **components)
    if coord_type is None:
        coord_type = input_type
    uses_names = (input_type == "names") or (coord_type == "names")
    if uses_names:
        if names is None:
            names = [(c,) for c in components]
        names = misc.assume_list(names)
        names = [misc.assume_tuple(c) for c in names]
        # Ensure first name option is used as component name
        if input_type == "names":
            names_map = {}
            for name_options in names:
                for name in name_options:
                    names_map[name] = name_options[0]
            components = {names_map[k]: v for k, v in components.items()}
    # If no conversion necessary
    if input_type == coord_type:
        if coord_type == "names":
            return components
        elif coord_type == "coords":
            return coords[0]
        else:
            return coords
    # Convert to temporary type "components"
    if input_type == "names":
        coords = [components[name_options[0]] for name_options in names]
    elif input_type == "coords":
        coords = np.moveaxis(coords[0], -1, 0)
    # Return data in converted type
    if coord_type == "names":
        if len(names) != len(coords):
            raise IndexError("Inconsistent number of components")
        return {
            name_options[0]: coords[i]
            for i, name_options in enumerate(names)
        }
    elif coord_type == "coords":
        return np.moveaxis(coords, 0, -1)
    elif coord_type == "components":
        return coords
    else:
        raise ValueError("Invalid `coord_type`")


def process_coords_factory(
    component_names, ret_names, coord_type="components"
):
    """
    Unifies the specification of coordinates.

    * Creates a decorator for functions that take coordinates as arguments.
    * The function to be decorated must take positional arguments
      corresponding to the coordinate components.
      It should return the converted coordinate components.
    * If a single (multidimensional) positional argument is given,
      the coordinate components are assumed to be located in the last axis.
    * If multiple (multidimensional) positional arguments are given,
      each argument is assumed to be a coordinate component.
    * Alternatively, keyword arguments can be given to directly specify
      the coordinate components.
      The parameter `component_names` is used to specify the component names.
    * The decorated function accepts the `ret_type` keyword argument, with
      which the return type (`"coords", "components", "names"`)
      can be specified. By default, the return type matches the input.

    Parameters
    ----------
    component_names : `list(tuple(str))`
        Component names in the order used by the function to be decorated.
        Each tuple contains aliases for the respective component.
    ret_names : `list(str)`
        Component names of the new coordinate system.
    coord_type : `str`
        Coordinate format required by the function to be decorated:
        `"coords", "components", "names"`.

    See also
    --------
    Conversion between coordinate formats: :py:func:`assume_coords_type`.
    """
    # Parse parameters
    component_names = misc.assume_list(component_names)
    component_names = [misc.assume_tuple(c) for c in component_names]
    ret_names = misc.assume_list(ret_names)
    if len(component_names) != len(ret_names):
        raise ValueError("Inconsistent number of components")

    # Define decorator
    def dec_process_coords(_func):
        # Decorated function
        def func(*coords, ret_type=None, **components):
            # Process input coordinates
            input_type = get_coords_type(*coords, **components)
            coord_input = assume_coords_type(
                *coords, **components,
                coord_type=coord_type, names=component_names
            )
            args, kwargs = [], {}
            if coord_type == "components":
                args = coord_input
            elif coord_type == "coords":
                args = [coord_input]
            elif coord_type == "names":
                kwargs.update(coord_input)
            # Execute function
            ret = _func(*args, **kwargs)
            # Process return values
            if ret_type is None:
                ret_type = input_type
            args, kwargs = [], {}
            if coord_type == "components":
                args = ret
            elif coord_type == "coords":
                args = [ret]
            elif coord_type == "names":
                kwargs.update(ret)
            ret = assume_coords_type(
                *args, **kwargs,
                coord_type=ret_type, names=ret_names
            )
            return ret
        # Return decorated function
        func.__doc__ = _func.__doc__
        return func
    # Return decorator
    return dec_process_coords


# ++++++++++++++++++++++++++++++++++++++++++++++++++
# Two-dimensional
# ++++++++++++++++++++++++++++++++++++++++++++++++++


@process_coords_factory(component_names=[
    ("r", "radial"),
    ("phi", "φ", "polar")
], ret_names=["x", "y"], coord_type="components")
def cv_polar_to_cartesian(r, phi):
    r"""
    Converts polar :math:`(\rho, \phi)` to cartesian coordinates.

    .. math:
        x = \rho \cdot \cos (\phi), \\
        y = \rho \cdot \sin (\phi),

    where :math:`r \in [0, \infty]` is the radial,
    :math:`\phi \in [0, 2 \pi]` is the polar coordinate.
    """
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return np.array((x, y))


@process_coords_factory(component_names=[
    ("x", "horizontal"),
    ("y", "vertical")
], ret_names=["r", "phi"], coord_type="components")
def cv_cartesian_to_polar(x, y):
    r"""
    Converts cartesian to polar :math:`(\rho, \phi)` coordinates.

    .. math:
        x = \rho \cdot \cos (\phi), \\
        y = \rho \cdot \sin (\phi),

    where :math:`r \in [0, \infty]` is the radial,
    :math:`\phi \in [0, 2 \pi]` is the polar coordinate.
    """
    r = np.linalg.norm((x, y), axis=0)
    phi = np.arctan2(y, x)
    return np.array((r, phi))


# ++++++++++++++++++++++++++++++++++++++++++++++++++
# Three-dimensional
# ++++++++++++++++++++++++++++++++++++++++++++++++++


@process_coords_factory(component_names=[
    ("r", "radial"),
    ("theta", "θ", "polar"),
    ("phi", "φ", "azimuthal")
], ret_names=["x", "y", "z"], coord_type="components")
def cv_spherical_to_cartesian(r, theta, phi):
    r"""
    Converts spherical :math:`(r, \theta, \phi)` to cartesian coordinates.

    .. math:
        x = r \cdot \sin (\theta) \cos (\phi), \\
        y = r \cdot \sin (\theta) \sin (\phi), \\
        z = r \cdot \cos (\theta),

    where :math:`r \in [0, \infty]` is the radial, :math:`\theta \in [0, \pi]`
    is the polar, and :math:`\phi \in [0, 2 \pi]` is the azimuthal coordinate.
    """
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array((x, y, z))


@process_coords_factory(component_names=[
    ("x", "horizontal"),
    ("y", "orthogonal"),
    ("z", "vertical")
], ret_names=["r", "theta", "phi"], coord_type="components")
def cv_cartesian_to_spherical(x, y, z):
    r"""
    Converts cartesian to spherical :math:`(r, \theta, \phi)` coordinates.

    .. math:
        x = r \cdot \sin (\theta) \cos (\phi), \\
        y = r \cdot \sin (\theta) \sin (\phi), \\
        z = r \cdot \cos (\theta),

    where :math:`r \in [0, \infty]` is the radial, :math:`\theta \in [0, \pi]`
    is the polar, and :math:`\phi \in [0, 2 \pi]` is the azimuthal coordinate.
    """
    r = np.linalg.norm((x, y, z), axis=0)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return np.array((r, theta, phi))


@process_coords_factory(component_names=[
    ("rho", "r", "radial"),
    ("phi", "φ", "polar"),
    ("z", "vertical")
], ret_names=["x", "y", "z"], coord_type="components")
def cv_cylindrical_to_cartesian(rho, phi, z):
    r"""
    Converts cylindrical :math:`(\rho, \phi, z)` to cartesian coordinates.

    .. math:
        x = \rho \cdot \cos (\phi), \\
        y = \rho \cdot \sin (\phi), \\
        z = z,

    where :math:`r \in [0, \infty]` is the in-plane radial,
    :math:`\phi \in [0, 2 \pi]` is the polar,
    and :math:`z \in [-\infty, \infty]` is the vertical coordinate.
    """
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return np.array((x, y, z))


@process_coords_factory(component_names=[
    ("x", "horizontal"),
    ("y", "orthogonal"),
    ("z", "vertical")
], ret_names=["rho", "phi", "z"], coord_type="components")
def cv_cartesian_to_cylindrical(x, y, z):
    r"""
    Converts cartesian to cylindrical :math:`(\rho, \phi, z)` coordinates.

    .. math:
        x = \rho \cdot \cos (\phi), \\
        y = \rho \cdot \sin (\phi), \\
        z = z,

    where :math:`r \in [0, \infty]` is the in-plane radial,
    :math:`\phi \in [0, 2 \pi]` is the polar,
    and :math:`z \in [-\infty, \infty]` is the vertical coordinate.
    """
    rho = np.linalg.norm((x, y), axis=0)
    phi = np.arctan2(y, x)
    return np.array((rho, phi, z))
