        
from core.geo.naming import names


def test_names():
    
    # usage
    names.lat
    names.lat.desc
    names.lat.minv
    names.lat.maxv
    
    # theses attributes should not have any max or min values
    assert names.rows.minv is None
    assert names.rows.maxv is None
    assert names.columns.minv is None
    assert names.columns.maxv is None
    
    assert isinstance(names.lon.minv, (int, float))
    assert isinstance(names.lon.maxv, (int, float))
    assert names.lon.minv < names.lon.maxv
