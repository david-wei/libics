import numpy as np
from itertools import permutations, chain
from math import factorial
import time

from libics.core.data.arrays import ArrayData


###############################################################################
# Helper functions
###############################################################################


def _slice_by_lengths(lengths, the_list):
    idx = 0
    for length in lengths:
        new = the_list[idx:idx + length]
        idx += length
        yield new


def _partition(number):
    return {
        (x,) + y for x in range(1, number)
        for y in _partition(number-x)
    } | {(number,)}


def _subgroups(my_list):
    partitions = _partition(len(my_list))
    permed = [
        set(permutations(each_partition, len(each_partition)))
        for each_partition in partitions
    ]
    for each_tuple in chain(*permed):
        yield list(_slice_by_lengths(each_tuple, my_list))


# Gets prefactor of permutation
def _get_partition(my_list, num_groups):
    # FIXME: highly inefficient
    filtered = []
    for perm in permutations(my_list, len(my_list)):
        for sub_group_perm in _subgroups(list(perm)):
            if len(sub_group_perm) == num_groups:
                # within each partition
                sorted_list = [sorted(i) for i in sub_group_perm]
                # by number of elements and first element of each partition
                sorted_list = sorted(sorted_list, key=lambda t: (len(t), t[0]))
                if sorted_list not in filtered:
                    filtered.append(sorted_list)
    return filtered


###############################################################################
# Get Correlator of order n
###############################################################################


def Correlate(_data, _ds, connected_cor=True):
    if len(_data.shape) == 2:
        _data = np.array([_data])
    Npic = _data.shape[0]
    _YSize = _data.shape[1]
    _XSize = _data.shape[2]
    _YStart = max([np.clip(i, 0, _YSize) for i in _ds[1::2]])
    _YStop = min([np.clip(_YSize+i, 0, _YSize) for i in _ds[1::2]])
    _XStart = max([np.clip(-i, 0, _XSize) for i in _ds[0::2]])
    _XStop = min([np.clip(_XSize-i, 0, _XSize) for i in _ds[0::2]])
    _dsFull = [0, 0]+_ds
    n = np.empty((int(len(_dsFull)/2), Npic, _YStop-_YStart, _XStop-_XStart))
    for i, pos in enumerate(range(0, len(_dsFull), 2)):
        n[i] = _data[
            :, _YStart-_dsFull[pos+1]:_YStop-_dsFull[pos+1],
            _XStart+_dsFull[pos]:_XStop+_dsFull[pos]
        ]
    if connected_cor is False:
        return np.nanmean(np.nanmean(np.prod(n, axis=0), axis=0))
    _gN = range(int(len(_dsFull)/2))
    Totalsum = 0
    for part in _gN:
        _coff = (-1)**(part)*factorial(part)
        _groups = _get_partition(_gN, part+1)
        for _group in _groups:
            _res = 1
            for _sub in _group:
                _res = _res*np.nanmean(np.prod(n[_sub], axis=0), axis=0)
            Totalsum = Totalsum+_coff*_res
    return np.nanmean(Totalsum)


###############################################################################
# Calculate the correlator for a given data set
###############################################################################


def cal_particle_particle_correl(
    data, connected=True, n_order=2, eval_area=[5, 5], eval_length=5
):
    """
    Calculates a multidimensional cross-correlation of order N (N_order).

    Parameters
    ----------
    data : `ArrayData`
        Input data: an array with N subarrays, each being the occupation
        within our lattice or an abitrary lattice with real numbers
    connected : `boolean`
        If True: calculation of connected correlator
        If True: <XY>-<X><Y>
        If False: then only the first part of the correlator will be calculated
        If False: <XY>
    n_order : `float`
        Order of correlator
    default_area : `float`
        evaluation area
    default_length : `float`
        length of additional dimension, if higher-order is used

    Examples
    --------
    Input array should be an array of real numbers.

    Example Array:

    Application: Particle-particle correlations.
    Array indicated site occupation ("1" = atom, "0" = hole).

    The presented array consists of 5 independent occupations with
    a system size of 2*14 lattice sites.

    array([[[1, 0, 1, 1, 0, 0, 0, 0, 1, 1],
        [0, 1, 0, 1, 0, 0, 0, 1, 1, 1]],

       [[1, 0, 1, 0, 1, 1, 0, 1, 0, 0],
        [1, 0, 0, 1, 1, 0, 1, 0, 0, 1]],

       [[1, 1, 0, 1, 0, 1, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 1, 1, 1, 0, 1]],

       [[1, 0, 1, 0, 1, 0, 0, 1, 0, 1],
        [1, 0, 0, 0, 1, 1, 1, 0, 1, 0]],

       [[0, 0, 1, 1, 1, 0, 1, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 1, 1, 1]]])
    """
    start_time = time.time()

    dim = {}
    for i in range(n_order):
        if i < 2:
            dim["dim"+str(i+1)] = eval_area[i]
        else:
            dim["dim"+str(i+1)] = eval_length

    # Prepare the grid for evalution
    G_list = [2*x+1 for x in dim.values()]
    G = np.zeros(G_list)

    GIdxs = [(var) for var in np.ndindex(G.shape)]
    CalSize = len(GIdxs)

    for check, GIdx in enumerate(GIdxs):
        ds = []
        for counter, center in enumerate(dim.values()):
            ds.append(GIdx[counter] - center)

        G[GIdx] = Correlate(data, ds, connected_cor=connected)
        check += 1
        print('Progress: ', round(check/CalSize*100, 2), ' %', end='\r')

    center = [x for x in dim.values()]

    if connected:
        G[tuple(center)] = np.nan
    else:
        G[tuple(center)] = 0.0
    print("Time --- %0.4f seconds ---" % (time.time() - start_time))

    G = ArrayData(G)

    for i in range(G.ndim):
        G.set_dim(i, center=0, step=1)

    return G


###############################################################################
# Get Standard Deviation by Bootstrapping
###############################################################################


def Correlate_std(
    _data, _ds, connected=True, bs=[5, 20], n_order=2,
    eval_area=[5, 5], eval_length=5
):
    """
    Calculates the standard deviation of the multidimensional
    cross-correlation for one coordinate _ds

    Parameters
    ----------
    data : `ArrayData`
        Input data: an array with N subarrays
    _ds : `float`
        Coordinate on which correlator is evaluated
    connected : `boolean`
        If True: calculation of connected correlator
        If True: <XY>-<X><Y>
        If False: then only the first part of the correlator will be calculated
        If False: <XY>
    bs : `int`
        bs is a list of two bootstrap parameters
        bs[0]: Number of groups
        bs[1]: Sample size per group
    n_order : `float`
        Order of correlator
    default_area : `float`
        evaluation area
    default_length : `float`
        length of additional dimension, if higher-order is used

    Examples
    --------
    Input array should be an array of real numbers.

    Example Array:

    Application: Particle-particle correlations.
    Array indicated site occupation ("1" = atom, "0" = hole).

    The presented array consists of 5 independent occupations with
    a system size of 2*14 lattice sites.

    array([[[1, 0, 1, 1, 0, 0, 0, 0, 1, 1],
        [0, 1, 0, 1, 0, 0, 0, 1, 1, 1]],

       [[1, 0, 1, 0, 1, 1, 0, 1, 0, 0],
        [1, 0, 0, 1, 1, 0, 1, 0, 0, 1]],

       [[1, 1, 0, 1, 0, 1, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 1, 1, 1, 0, 1]],

       [[1, 0, 1, 0, 1, 0, 0, 1, 0, 1],
        [1, 0, 0, 0, 1, 1, 1, 0, 1, 0]],

       [[0, 0, 1, 1, 1, 0, 1, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 1, 1, 1]]])
    """
    _group = bs[0]
    _n = bs[1]
    g_list = []
    _rand = np.random.randint(_data.shape[0], size=(_group, _n))
    for i, pos in enumerate(_rand):
        print(r'progress: {:0.2f} %'.format((i+1)/_group*100), end='\r')
        g_list.append(Correlate(_data[pos], _ds, connected_cor=connected))
    g_list = np.array(g_list)
    return np.std(g_list)/np.sqrt(_group)


###############################################################################
# Calculate the standard error of the mean by bootstrapping
###############################################################################


def cal_particle_particle_correl_std(
    data, connected=True, bs=[5, 20], n_order=2,
    eval_area=[5, 5], eval_length=5
):
    """
    Calculates the standard deviation of the multidimensional cross-correlation
    of order N (N_order) using bootstrapping.

    Parameters
    ----------
    data : `ArrayData`
        Input data: an array with N subarrays
    _ds : `float`
        Coordinate on which correlator is evaluated
    connected : `boolean`
        If True: calculation of connected correlator
        If True: <XY>-<X><Y>
        If False: then only the first part of the correlator will be calculated
        If False: <XY>
    bs : `int`
        bs is a list of two bootstrap parameters
        bs[0]: Number of groups
        bs[1]: Sample size per group
    n_order : `float`
        Order of correlator
    default_area : `float`
        evaluation area
    default_length : `float`
        length of additional dimension, if higher-order is used

    Examples
    --------
    Input array should be an array of real numbers.

    Example Array:

    Application: Particle-particle correlations.
    Array indicated site occupation ("1" = atom, "0" = hole).

    The presented array consists of 5 independent occupations with
    a system size of 2*14 lattice sites.

    array([[[1, 0, 1, 1, 0, 0, 0, 0, 1, 1],
        [0, 1, 0, 1, 0, 0, 0, 1, 1, 1]],

       [[1, 0, 1, 0, 1, 1, 0, 1, 0, 0],
        [1, 0, 0, 1, 1, 0, 1, 0, 0, 1]],

       [[1, 1, 0, 1, 0, 1, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 1, 1, 1, 0, 1]],

       [[1, 0, 1, 0, 1, 0, 0, 1, 0, 1],
        [1, 0, 0, 0, 1, 1, 1, 0, 1, 0]],

       [[0, 0, 1, 1, 1, 0, 1, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 1, 1, 1]]])

    """

    start_time = time.time()

    dim = {}
    for i in range(n_order):
        if i < 2:
            dim["dim"+str(i+1)] = eval_area[i]
        else:
            dim["dim"+str(i+1)] = eval_length

    # Prepare the grid for evalution
    G_list = [2*x+1 for x in dim.values()]
    G = np.zeros(G_list)

    GIdxs = [(var) for var in np.ndindex(G.shape)]
    CalSize = len(GIdxs)

    for check, GIdx in enumerate(GIdxs):
        ds = []
        for counter, center in enumerate(dim.values()):
            ds.append(GIdx[counter] - center)

        G[GIdx] = Correlate_std(data, ds, connected=connected, bs=bs)
        check += 1
        print('Progress: ', round(check/CalSize*100, 2), ' %', end='\r')

    center = [x for x in dim.values()]

    if connected:
        G[tuple(center)] = 0.0
    else:
        G[tuple(center)] = np.nan
    print("Time --- %0.4f seconds ---" % (time.time() - start_time))

    G = ArrayData(G)

    for i in range(len(G.shape)):
        G.set_dim(i, center=0, step=1)

    return G
