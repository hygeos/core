# standard library imports
# ...
        
# third party imports
import xarray as xr
import numpy as np
        
# sub package imports
from core.naming import *


def test_names():
    
    # usage
    names.latitude.name
    names.latitude.desc
    names.latitude.minv
    names.latitude.maxv
    
    # theses attributes should not have any max or min values
    assert names.rows.minv is None
    assert names.rows.maxv is None
    assert names.columns.minv is None
    assert names.columns.maxv is None
    
    assert type(names.longitude.minv) in (int|float).__args__
    assert type(names.longitude.maxv) in (int|float).__args__
    assert names.longitude.minv < names.longitude.maxv

def test_var_addition():
    ds = xr.Dataset()
    da = xr.DataArray(np.zeros((10,10)))
    add_var(ds, da, names.lon)