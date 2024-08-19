# standard library imports
# ...
        
# third party imports
# ...
        
# sub package imports
from core.naming import names


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
    