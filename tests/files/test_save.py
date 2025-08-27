
import pytest
from pathlib import Path
from tempfile import TemporaryDirectory

from core.files.save import to_netcdf
import xarray as xr


def ds_example(): return xr.Dataset({'test': range(10)})
def da_example(): return xr.DataArray(range(10))


def test_to_netcdf():
    with TemporaryDirectory() as tmpdir:
        to_netcdf(ds_example(), Path(tmpdir)/'test.nc')

def test_to_netcdf_format_check(tmpdir):
    with TemporaryDirectory() as tmpdir:
        with pytest.raises(AssertionError):
            to_netcdf('inp', Path(tmpdir)/'test.nc')

def test_to_netcdf_format_check_da(tmpdir):
    with TemporaryDirectory() as tmpdir:
        with pytest.raises(AssertionError):
            to_netcdf(da_example(), Path(tmpdir)/'test.nc')