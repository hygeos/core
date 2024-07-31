import pytest

from core.static import frozen


def test_set_attr():
    
    @frozen
    class Object:
        def __init__(self):
            self.a = 1
            self.b = 2

    a = Object()
    
    # can modify attributes in place
    a.a = "a"
    a.b = "b"

    # cannot create new attribute outside constructor 
    with pytest.raises(Exception):
        a.c = "c"
            

def test_subclasses():
    
    @frozen
    class Object:
        def __init__(self):
            self.a = 1
            self.b = 2
    
    class Thing(Object):
        pass
    
    a = Thing()
    
    with pytest.raises(Exception):
        a.c = "c"
    
    with pytest.raises(Exception):
        a._frozen = False


def test_wrong_usage():
    
    with pytest.raises(Exception):
        @frozen
        def function(x):
            return x * x
