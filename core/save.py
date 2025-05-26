#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from contextlib import nullcontext
from pathlib import Path
from typing import Literal, Optional

import xarray as xr
from dask.diagnostics.progress import ProgressBar

from core.fileutils import filegen
from core import log

def to_netcdf(
    ds: xr.Dataset,
    filename: Path,
    *,
    engine: str = "h5netcdf",
    zlib: bool = True,
    complevel: int = 5,
    verbose: bool = True,
    tmpdir: Optional[Path] = None,
    lock_timeout: int = 0,
    if_exists: Literal["skip", "overwrite", "backup", "error"] = "error",
    clean_attrs: bool = True,
    **kwargs
):
    """
    Write an xarray Dataset `ds` using `.to_netcdf` with several additional features:

    - Use file compression
    - Wrapped by filegen: use temporary files, detect existing output files...

    Args:
        ds (xr.Dataset): Input dataset
        filename (Path): Output file path
        engine (str, optional): Engine driver to use. Defaults to 'h5netcdf'.
        zlib (bool, optional): activate zlib. Defaults to True.
        complevel (int, optional): Compression level. Defaults to 5.
        verbose (bool, optional): Verbosity. Defaults to True.
        tmpdir (Path, optional): use a given temporary directory. Defaults to None.
        lock_timeout (int): timeout in case of existing lock file
        if_exists (str, optional): what to do if output file exists. Defaults to 'error'.
        clean_attrs: whether to remove attributes in the xarray object, that cannot
            be written to netcdf.
        other kwargs are passed to ds.to_netcdf
    """
    soluce = ', got an xarray DataArray. Please use .to_dataset method and ' \
             'specify a variable name' if isinstance(ds, xr.DataArray) else ''
    if not isinstance(ds, xr.Dataset):
        log.error("to_netcdf expects an xarray Dataset" + soluce, 
                  e=AssertionError)

    encoding = (
        {var: dict(zlib=True, complevel=complevel) for var in ds.data_vars}
        if zlib
        else None
    )

    if clean_attrs:
        clean_attributes(ds)

    PBar = {True: ProgressBar, False: nullcontext}[verbose]

    with PBar():
        if verbose:
            log.info("Writing:", filename)

        filegen(
            0,
            tmpdir=tmpdir,
            lock_timeout=lock_timeout,
            if_exists=if_exists,
            verbose=verbose,
        )(ds.to_netcdf)(filename, engine=engine, encoding=encoding, **kwargs)


def clean_attributes(obj: xr.Dataset|xr.DataArray):
    """
    Remove attributes that can not be written to netcdf
    """
    import numpy as np
    for attr in list(obj.attrs):
        if isinstance(obj.attrs[attr], (bool,)):
            obj.attrs[attr] = str(obj.attrs[attr])
        elif not isinstance(obj.attrs[attr], (str, float, int, np.ndarray, np.number)):
            del obj.attrs[attr]
    
    # recursively clean attributes in the individual variables
    if isinstance(obj, xr.Dataset):
        for var in list(obj):
            clean_attributes(obj[var])
