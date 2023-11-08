"""Module providing multi-axes plots of higher-dimensional data."""

import numpy as np

from libics.env.logging import get_logger
from libics.core.data.sequences import DataSequence
from libics.tools.plot.base import remove_axes, subplots, unsqueeze_axes

LOGGER = get_logger("libics.tools.plot.multi")


###############################################################################


# Predefined select plot parameters
_DEFAULT_KWARGS = {"row", "col", "marker", "linestyle", "color"}
_MARKERS = ["[o]", "[^]", "[s]", "[X]", "[D]", "[+]", "[*]", "[H]"]
_LINESTYLES = ["solid", "dotted", "dashed", "dashdot"]
_COLORS = [f"C{i:d}" for i in range(10)]
# Map plot parameter abbreviations
_PLT_PARAM_MAP = {"lw": "linewidth"}


def _analyze_dataset_for_ax_array(
    dataset, arg_keys, kwarg_keys=None, select_keys=None, x_key=None
):
    """
    Creates a data set containing data prepared for plotting an axes array.

    Parameters
    ----------
    dataset : `pd.DataFrame`
    arg_keys : `Iter[str]`
    kwarg_keys : `dict(str->str)`
    select_keys : `dict(str->Iter[str])`
    x_key : `str`

    Returns
    -------
    res : `dict(str->Any)` containing
    data : `DataSequence`
    arg_size : `int`
    kwargs_list : `list(str)`
    select_list : `list(str)`
    x_key : `bool`
    """
    # Parse parameters
    if isinstance(arg_keys, str):
        arg_keys = [arg_keys]
    if kwarg_keys is None:
        kwarg_keys = {}
    if select_keys is None:
        select_keys = {}
    else:
        select_keys = select_keys.copy()
    # Prepare ax array data set
    dsaa = DataSequence()
    for i, arg_key in enumerate(arg_keys):
        dsaa[f"arg_{i:d}"] = dataset[arg_key]
    for k, kwarg_key in kwarg_keys.items():
        dsaa[f"kwarg_{k}"] = dataset[kwarg_key]
    for s, select_key in select_keys.items():
        if isinstance(select_key, str):
            select_key = [select_key]
        dsaa[f"select_{s}"] = [
            tuple(ds_row[sk] for sk in select_key)
            for _, ds_row in dataset.iterrows()
        ]
    # Set additional item for 1D plots
    if x_key is not None:
        dsaa["arg_x"] = dataset[x_key]
    # Ensure row and col are selected
    for k in ["row", "col"]:
        if k not in select_keys:
            select_keys[k] = [(None,)]
            dsaa[f"select_{k}"] = len(dsaa) * [(None,)]
    return {
        "data": dsaa, "arg_size": len(arg_keys),
        "kwargs_list": list(kwarg_keys), "select_list": list(select_keys),
        "x_key": x_key is not None
    }


def _analyze_shared_keys_for_ax_array(dsaa, select_list, share_list):
    """
    Parameters
    ----------
    dsaa : `pd.DataFrame`
    select_list : `Iter[str]`
    share_list : `Iter[str]`

    Returns
    -------
    res : `dict(str->Any)` containing
    nrows, ncols : `int`
    unique_vals : `dict(str->list(Any))`
    unshared_vals : `dict(str->list(list))`
    """
    # Get unique values by selected keys
    unique_vals = {
        select_key: sorted(set(dsaa[f"select_{select_key}"]))
        for select_key in select_list
    }
    # Variable for unshared row or col values
    unshared_vals = {}
    # Obtain nrows/ncols from shared keys
    nrows, ncols = None, None
    if "row" in share_list:
        nrows = len(unique_vals["row"])
    if "col" in share_list:
        ncols = len(unique_vals["col"])
    # If both row and col are not provided in share_list,
    # at least one of them must be shared: Here we default to row.
    if nrows is None and ncols is None:
        nrows = len(unique_vals["row"])
    # Find nrows/ncols if not shared
    if nrows is None or ncols is None:
        shared_ax = "row" if nrows is not None else "col"
        unshared_ax = "row" if nrows is None else "col"
        unshared_vals[unshared_ax] = [
            sorted(set(
                dsaa[dsaa[f"select_{shared_ax}"] == k][f"select_{unshared_ax}"]
            )) for i, k in enumerate(unique_vals[shared_ax])
        ]
        unshared_size = np.max([len(v) for v in unshared_vals[unshared_ax]])
        if nrows is None:
            nrows = unshared_size
        elif ncols is None:
            ncols = unshared_size
    return {
        "nrows": nrows, "ncols": ncols,
        "unique_vals": unique_vals, "unshared_vals": unshared_vals
    }


def _plot_ax_in_ax_array(
    dsaa, plot_func, ax,
    arg_size, kwargs_list, select_list, share_list, unique_vals,
    x_key=False, quantitative_list=None, fmt_keys=None,
    **kwargs
):
    """
    Perform multiple plots in one matplotlib axes.

    Parameters
    ----------
    dsaa : `pd.DataFrame`
        Data set analyzed by :py:func:`_analyze_dataset_for_ax_array`.
    plot_func : `callable`
        Plot function. Must accept parameter `ax` as matplotlib axes.
    ax : `matplotlib.axes.Axes`
        Matplotlib axes to plot in.
    arg_size : `int`
        Number of positional arguments for `plot_func`.
    kwargs_list : `Iter[str]`
        Names of keyword arguments for `plot_func`.
    select_list : `Iter[str]`
        List of plot keys selected for data differentiation.
    share_list : `Iter[str]`
        List of plot keys which should have common representation across axes.
    unique_vals : `dict(str->list)`
        Globally unique values for each plot key.
    x_key : `bool`
        Whether to plot 1D dependence.
    quantitative_list : `Iter[str]` or `None`
        List of plot keys to be used quantitatively.
    fmt_keys : `dict(str->str or True)`
        Formatting string by plot key.
    **kwargs : `str->Any` or `str->Iter[Any]`
        Keyword arguments for plot keys, e.g., `color="red"`.
        If the plot key is selected for data differentiation, append an `s`,
        e.g., `colors=["red", "green", "blue"]`.
    """
    # Parse parameters
    select_list = select_list.copy()
    for k in ["row", "col"]:
        if k in select_list:
            select_list.remove(k)
    if quantitative_list is None:
        quantitative_list = []
    if fmt_keys is None:
        fmt_keys = {}
    # Default kwargs
    default_kwargs = dict(
        colors=_COLORS, markers=_MARKERS, linestyles=_LINESTYLES
    )
    for select_key in select_list:
        if f"{select_key}s" not in kwargs and select_key not in fmt_keys:
            kwargs[f"{select_key}s"] = default_kwargs[f"{select_key}s"]
    # Get variable plotting kwargs
    select_kwargs = {
        select_key: (
            None if select_key in fmt_keys else kwargs.pop(f"{select_key}s")
        ) for select_key in select_list
    }
    # Get local unique vals for unshared keys
    local_unique_vals = {
        select_key: (
            unique_val if select_key in share_list
            else sorted(set(dsaa[f"select_{select_key}"]))
        ) for select_key, unique_val in unique_vals.items()
    }
    # Plot data
    if x_key is False:
        for _, ds_row in dsaa.iterrows():
            func_args = [ds_row[f"arg_{i:d}"] for i in range(arg_size)]
            func_kwargs = {k: ds_row[f"kwarg_{k}"] for k in kwargs_list}
            plt_params = {}
            for select_key in select_list:
                select_val = ds_row[f"select_{select_key}"]
                plt_vals = select_kwargs[select_key]
                if select_key in quantitative_list:
                    param = plt_vals(select_val)
                elif select_key in fmt_keys:
                    if fmt_keys[select_key] is True:
                        if len(select_val) == 1:
                            param = str(select_val[0])
                        else:
                            param = str(select_val)
                    else:
                        param = fmt_keys[select_key].format(*select_val)
                else:
                    _idx = local_unique_vals[select_key].index(select_val)
                    param = plt_vals[_idx % len(plt_vals)]
                plt_params[select_key] = param
            plot_func(*func_args, **func_kwargs, ax=ax, **plt_params, **kwargs)
    else:
        ds = dsaa.copy()
        ds["unique_select_comb"] = [
            tuple(ds_row[f"select_{select_key}"] for select_key in select_list)
            for _, ds_row in ds.iterrows()
        ]
        unique_select_combs = sorted(set(ds["unique_select_comb"]))
        for unique_select_comb in unique_select_combs:
            # Filter unique selection combination
            _ds = ds[ds["unique_select_comb"] == unique_select_comb].copy()
            _ds.sort_rows("arg_x")
            func_args = [_ds[f"arg_{i:d}"] for i in range(arg_size)]
            func_kwargs = {k: _ds[f"kwarg_{k}"] for k in kwargs_list}
            # Prepare plot parameters
            ds_row = _ds.iloc[0]
            plt_params = {}
            for select_key in select_list:
                select_val = ds_row[f"select_{select_key}"]
                plt_vals = select_kwargs[select_key]
                if select_key in quantitative_list:
                    param = plt_vals(select_val)
                elif select_key in fmt_keys:
                    if fmt_keys[select_key] is True:
                        if len(select_val) == 1:
                            param = str(select_val[0])
                        else:
                            param = str(select_val)
                    else:
                        param = fmt_keys[select_key].format(*select_val)
                else:
                    _idx = local_unique_vals[select_key].index(select_val)
                    param = plt_vals[_idx % len(plt_vals)]
                plt_params[select_key] = param
            plot_func(
                _ds["arg_x"], *func_args, **func_kwargs,
                ax=ax, **plt_params, **kwargs
            )


def plot_ax_array(
    # Data
    dataset, plot_func,
    # Grouping properties
    x_key=None, arg_keys=None, kwarg_keys=None, select_keys=None,
    # Figure properties
    fig=None, axs=None, sharex=False, sharey=False, remove_empty=True,
    figsize=None, axsize=None, axsize_offset=(0.2, 0.2), size_unit="in",
    # Formatting properties
    share_list=None, quantitative_list=None, fmt_keys=None, **plt_params
):
    """
    Plots a data set into a matplotlib axes array.

    Different data dimensions can be encoded in the plot style.

    Parameters
    ----------
    dataset : `pd.DataFrame` or `dict(str->list)`
        Data set.
    plot_func : `callable`
        Plot function. Must accept parameter `ax` as matplotlib axes.
    x_key : `str`
        If provided, the plot call is altered. After filtering the data set
        according to `select_keys`, `plot_func` is called as:
        `plot_func(dataset[x_key], dataset[arg_keys[0]], ...)`
    arg_keys : `Iter[str]`
        List of data keys used to call `plot_func`.
        Call signature: `plot_func(dataset_row[arg_key[0]], ...)`.
    kwarg_keys : `dict(str->str)`
        Dictionary mapping keyword to data keys used to call `plot_func`.
        Call signature for `key->val`:
        `plot_func(..., key=dataset_row[val], ...)`.
    select_keys : `dict(str->Iter[str])`
        Chooses how the data should be encoded as plot styles.
        Dictionary mapping plot style to data keys, e.g.,
        `{"color": ["data_column0", "data_column1"]}`.
    fig, axs
        Matplotlib figure or axes into which to plot data.
        If `None`, is automatically generated.
    sharex, sharey : `bool`
        Whether to share axes ticks.
    remove_empty : `bool`
        Whether to remove empty matplotlib axes.
    fig_size, ax_size, axsize_offset, size_unit
        Figure size if `fig` is automatically generated.
    share_list : `Iter[str]` or `None`
        List of plot keys which should have common representation across axes.
    quantitative_list : `Iter[str]` or `None`
        List of plot keys to be used quantitatively.
    fmt_keys : `dict(str->str or True)`
        Formatting string by plot key. Use `"{0}, {1}, ..."` to select
        the argument as ordered in `select_keys`.
    **plt_params : `str->Any` or `str->Iter[Any]`
        Keyword arguments for plot keys, e.g., `color="red"`.
        If the plot key is selected for data differentiation, append an `s`,
        e.g., `colors=["red", "green", "blue"]`.

    Returns
    -------
    axs : `np.ndarray(2, matplotlib.axes.Axes)`
        Matplotlib axes array.

    Examples
    --------
    Simple example for plotting two rows of 1D plots with different colors:

    >>> dataset = DataSequence({
    ...     "x": np.arange(10),
    ...     "y": np.linspace(-1, 1, num=10),
    ...     "z": 5*["a"] + 5*["b"],
    ...     "a": 5*[True] + 3*[False] + 2*[True],
    ... })
    >>> arg_keys = ["y"]
    >>> select_keys = {"color": "a", "label": "a", "row": "z"}
    >>> share_list = []
    >>> fmt_keys = {"label": r"a = {0}"}
    >>> x_key = "x"
    >>> axs = plot.plot_ax_array(
    ...     dataset, plot.scatter,
    ...     x_key=x_key, arg_keys=arg_keys,
    ...     select_keys=select_keys,
    ...     share_list=share_list, fmt_keys=fmt_keys,
    ...     axsize=(5, 3.5)
    ... )
    >>> for ax in np.ravel(axs):
    ...     plot.style_axes(ax=ax, legend=True)
    >>> plot.show()
    """
    # Parse and check parameters
    dataset = DataSequence(dataset)
    if arg_keys is None:
        arg_keys = []
        if np.any([x in plt_params for x in [
            "args_keys", "args_key", "arg_key"
        ]]):
            LOGGER.warn("Did you mean the parameter `arg_keys`?")
    elif isinstance(arg_keys, str):
        arg_keys = [arg_keys]
    if kwarg_keys is None:
        kwarg_keys = {}
        if np.any([x in plt_params for x in [
            "kwargs_keys", "kwargs_key", "kwarg_key"
        ]]):
            LOGGER.warn("Did you mean the parameter `kwarg_keys`?")
    if share_list is None:
        share_list = []
    elif isinstance(share_list, str):
        share_list = [share_list]
    if isinstance(quantitative_list, str):
        quantitative_list = [quantitative_list]
    for kk, vv in _PLT_PARAM_MAP.items():
        # Iterate singular (const plot param.) and plural (select plot param.)
        for k, v in [(kk, vv), (f"{kk}s", f"{vv}s")]:
            if k in plt_params:
                tmp = plt_params.pop(k)
                if v not in plt_params:
                    plt_params[v] = tmp
        # Only singular in select_keys
        if select_keys is not None:
            if kk in select_keys:
                tmp = select_keys.pop(kk)
                if vv not in select_keys:
                    select_keys[vv] = tmp
    if select_keys is not None:
        for k in select_keys:
            if (
                k not in _DEFAULT_KWARGS
                and f"{k}s" not in plt_params
                and k not in fmt_keys
            ):
                raise ValueError(f"`plt_param` parameter `{k}s` is required")

    # Analyze data set
    ana_ds = _analyze_dataset_for_ax_array(
        dataset, arg_keys, kwarg_keys=kwarg_keys,
        select_keys=select_keys, x_key=x_key
    )
    dsaa = ana_ds["data"]
    select_list = ana_ds["select_list"]
    # Analyze shared keys
    ana_sk = _analyze_shared_keys_for_ax_array(dsaa, select_list, share_list)
    nrows, ncols = ana_sk["nrows"], ana_sk["ncols"]
    unique_vals = ana_sk["unique_vals"]
    has_unshared_ax = len(ana_sk["unshared_vals"]) > 0
    if has_unshared_ax:
        unshared_ax = list(ana_sk["unshared_vals"].keys())[0]
        shared_ax = "row" if unshared_ax == "col" else "col"
        unshared_vals = ana_sk["unshared_vals"][unshared_ax]

    # Set up axes
    if axs is None:
        fig, axs = subplots(
            fig=fig, size_unit=size_unit, figsize=figsize,
            axsize=axsize, axsize_offset=axsize_offset,
            nrows=nrows, ncols=ncols, sharex=sharex, sharey=sharey,
            squeeze=False
        )
    else:
        ax_1d = "row" if ncols == 1 else "col"
        axs = unsqueeze_axes(axs, ax_1d=ax_1d)
        fig = axs[0, 0].get_figure()

    # Iterate axes for plotting
    empty_axs = []
    if not has_unshared_ax:
        for row, col in np.ndindex(nrows, ncols):
            ax = axs[row, col]
            row_val = unique_vals["row"][row]
            col_val = unique_vals["col"][col]
            _ds = dsaa[
                (dsaa["select_row"] == row_val)
                & (dsaa["select_col"] == col_val)
            ]
            if len(_ds) == 0:
                if remove_empty:
                    empty_axs.append(ax)
                    axs[row, col] = None
                continue
            _plot_ax_in_ax_array(
                _ds, plot_func, ax,
                ana_ds["arg_size"], ana_ds["kwargs_list"],
                ana_ds["select_list"], share_list,
                ana_sk["unique_vals"], x_key=ana_ds["x_key"],
                quantitative_list=quantitative_list, fmt_keys=fmt_keys,
                **plt_params
            )
    else:
        for row, col in np.ndindex(nrows, ncols):
            ax = axs[row, col]
            if shared_ax == "row":
                shared_idx, unshared_idx = row, col
            else:
                shared_idx, unshared_idx = col, row
            shared_val = unique_vals[shared_ax][shared_idx]
            try:
                unshared_val = unshared_vals[shared_idx][unshared_idx]
            except IndexError:
                empty_axs.append(ax)
                continue
            _ds = dsaa[
                (dsaa[f"select_{shared_ax}"] == shared_val)
                & (dsaa[f"select_{unshared_ax}"] == unshared_val)
            ]
            if len(_ds) == 0:
                if remove_empty:
                    empty_axs.append(ax)
                    axs[row, col] = None
                continue
            _plot_ax_in_ax_array(
                _ds, plot_func, ax,
                ana_ds["arg_size"], ana_ds["kwargs_list"],
                ana_ds["select_list"], share_list,
                ana_sk["unique_vals"], x_key=ana_ds["x_key"],
                quantitative_list=quantitative_list, fmt_keys=fmt_keys,
                **plt_params
            )
    # Clean up
    if remove_empty:
        remove_axes(*empty_axs, enforce=False, on_empty=True)
    return axs
