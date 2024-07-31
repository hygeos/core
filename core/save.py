#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Various utility functions for saving results
'''

import shutil
import tempfile
import xarray as xr
from typing import Union
from pathlib import Path
from contextlib import contextmanager
# from dask.diagnostics import ProgressBar
from typing import Literal

def to_netcdf(ds: xr.Dataset, filename: Union[str, Path],
              *,
              output_dir: Union[str, Path],
              ext: str = '.nc',
              if_exists: str = Literal['error', 'skip', 'overwrite'],
              tmpdir: str = None,
              create_out_dir: bool = True,
              engine: str = 'h5netcdf',
              zlib: bool = True,
              complevel: int = 5,
              verbose: bool = True,
              **kwargs):
    """
    Write an xarray Dataset `ds` using `.to_netcdf` with several additional features:
    - construct file name using  `dirname`, `product_name` and `ext`
      (optionnally - otherwise with `filename`)
    - check that the output file does not exist already
    - Use file compression
    - Use temporary file
    - Create output directory if it does not exist

    Args:
        ds (xr.Dataset): _description_
        filename (Union[str, Path], optional): Output file path, if None, build filename from `dirname`, `product_name` and `ext`. Defaults to None.
        dirname (Union[str, Path], optional): directory for output file (default None: uses the attribute input_directory of ds). Defaults to None.
        product_name (str, optional): base name for the output file. If None (default), use the attribute named attr_name. Defaults to None.
        ext (str, optional): extension. Defaults to '.nc'.
        product_name_attr (str, optional): name of the attribute to use for product_name in `ds`. Defaults to 'product_name'.
        if_exists (str, optional): what to do if output file exists. Defaults to 'error'.
        tmpdir (str, optional): use a given temporary directory. Defaults to None.
        create_out_dir (bool, optional): create output directory if it does not exist. Defaults to True.
        engine (str, optional): Engine driver to use. Defaults to 'h5netcdf'.
        zlib (bool, optional): _description_. Defaults to True.
        complevel (int, optional): Compression level. Defaults to 5.
        verbose (bool, optional): _description_. Defaults to True.

    Raises:
        IOError: _description_
        ValueError: _description_
        IOError: _description_

    Returns:
        str: output file name
    """
    assert isinstance(ds, xr.Dataset), 'to_netcdf expects an xarray Dataset'
    
    output_dir = Path(output_dir)
    if create_out_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        assert output_dir.is_dir(), f"please create output directory {output_dir}, or allow automatic output folder creation"
    
    fname = (output_dir / Path(str(filename) + ext)).resolve()

    if fname.exists():
        if if_exists == 'skip':
            print(f'File {fname} already exists, skipping...')
            return fname
        elif if_exists == 'overwrite':
            fname.unlink()
        elif if_exists == 'error':
            raise IOError(f'Output file "{fname}" exists.')
        else:
            raise ValueError(f'Invalid option for "if_exists": {if_exists}')

    encoding = {var: dict(zlib=True, complevel=complevel)
                for var in ds.data_vars} if zlib else None

    # PBar = {
        # True: ProgressBar,
        # False: none_context
    # }[verbose]

    # with PBar(), tempfile.TemporaryDirectory(dir=tmpdir) as tmp:
    with tempfile.TemporaryDirectory(dir=tmpdir) as tmp:

        fname_tmp = Path(tmp)/fname.name

        if verbose:
            print('Writing:', fname)
            print('Using temporary file:', fname_tmp)

        ds.to_netcdf(path=fname_tmp,
                     engine=engine,
                     encoding=encoding,
                     **kwargs)

        # use intermediary move
        # (both files may be on different devices)
        shutil.move(fname_tmp, str(fname)+'.tmp')
        shutil.move(str(fname)+'.tmp', fname)

    return fname

def none_context(a=None):
    """
    Returns a context manager that does nothing.

    In python 3.7, this is equivalent to `contextlib.nullcontext`.
    """
    return contextmanager(lambda: (x for x in [a]))()

