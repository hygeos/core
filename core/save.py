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
        other kwargs are passed to ds.to_netcdf
    """
    assert isinstance(ds, xr.Dataset), "to_netcdf expects an xarray Dataset"

    encoding = (
        {var: dict(zlib=True, complevel=complevel) for var in ds.data_vars}
        if zlib
        else None
    )

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
