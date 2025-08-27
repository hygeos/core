# standard library imports
# ...
        
# third party imports
import xarray as xr
import numpy as np
        
# sub package imports
from core.geo.naming import names, add_var


def test_names():
    
    # usage
    names.lat.name
    names.lat.desc
    names.lat.minv
    names.lat.maxv
    
    # theses attributes should not have any max or min values
    assert names.rows.minv is None
    assert names.rows.maxv is None
    assert names.columns.minv is None
    assert names.columns.maxv is None
    
    assert type(names.lon.minv) in (int|float).__args__
    assert type(names.lon.maxv) in (int|float).__args__
    assert names.lon.minv < names.lon.maxv

def test_var_addition():
    ds = xr.Dataset()
    da = xr.DataArray(np.zeros((10,10)))
    add_var(ds, da, names.lon)