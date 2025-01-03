import pytest

from core.static import interface
from core.static.Exceptions import InterfaceException


def test_base():
    
    @interface # define function with typed parameters
    def function(a: int): return
    
    function(123)
    
    with pytest.raises(InterfaceException):
        function("a")

def test_default_none():
    
    @interface
    def function(a: int=None): return
    
    function(123)
    function()
    
    class obj:
        @interface
        def __init__(self, name:str=None): return
        
    a = obj("a")
    b = obj()

def test_mix():
    @interface # define function with typed parameters
    def function(a, b: int, c, d: str): return
    
    function("a", 123, 3.14, "test")    
    
    with pytest.raises(InterfaceException): 
        function("a", "b", 3.14, "test")   # invalid second parameter
        function("a", 123, 3.14, 1234)     # invalid fourth parameter
        
def test_unpack():
    @interface # define function with typed parameters
    def function(a, b: int, c, d: str): return
    
    l = [123, 3.14, "test"]
    function(True, *l)
    
    with pytest.raises(InterfaceException):
        function(*l, True)
    
def test_named():
    
    @interface # define function with typed parameters
    def function(a, b: int, c: float, d: str): return
    
    l = [123, 3.14]
    function(True, *l, d="test")


def test_default():
    
    @interface # define function with typed parameters
    def function(a, b: int, c: float = 3.14, d: str = "none"): 
        print(a,b,c,d); return
    
    l = [123]
    function(True, *l, d="test")
    
    
def test_container():
    
    @interface # define function with typed parameters
    def function(a, b: int, c: float = 3.14, d: list[str] = []): 
        print(a,b,c,d); return
    
    l = [123]
    function(True, *l, d=[1]) 
    # warning, the @interface wrapper is not recursive (yet)
    # so subtype, inside containers for example, are not checked
    
def test_kwargs():
    
    @interface # define function with typed parameters
    def function(a, b: int, c: float = 3.14, d: list[str] = []): 
        print(a,b,c,d); return
        
    kw = dict(b=123, c=3.14)
    function(True, **kw, d=[])
    
    
def test_subclass():
    
    from datetime import date, datetime
    
    @interface # define function with typed parameters
    def function(day: date): return
    
    # valid calls
    function(date(1999, 8, 24))
    function(datetime(1999, 8, 24, 14, 35, 11)) # valid because datetime is a subclass of date
    
    @interface # define function with typed parameters
    def function2(dt: datetime): return
    
    # invalid call (date is not a subclass of datetime)
    with pytest.raises(InterfaceException):
        function2(date(1999, 8, 24))


def test_kwargs_default():
    
    @interface # define function with typed parameters
    def function(a, b: int, c: float=3.14, d: list=[], e: bool=True): 
        return
        
    kw = dict(b=123, c=3.14)
    function(True, **kw, e=True)
    
    with pytest.raises(InterfaceException):
        kw2 = dict(b=123, c=3)      
        function(True, **kw2, e=3.14) # e isn't bool, should fail
    
    with pytest.raises(InterfaceException): # check that types are checked from unpacked kwargs
        kw3 = dict(b=123, c=3)      
        function(True, **kw3, e=False) # c isn't float, should fail


def test_kwargs_wrong_missed_default():
    
    @interface # define function with typed parameters
    def function(a, b: int, c: float=3.14, d: list={}, e: bool=True): 
        return
    
    with pytest.raises(InterfaceException):
        kw = dict(b=123, c=3.14)
        function(True, **kw, e=True)


def test_explicit_nonepassing():
    
    @interface
    def function(b, a: int=None, c: float=None): return
    
    a = function(1, None, c=None)
    
    
def test_unallowed_explicit_nonepassing():
    
    with pytest.raises(InterfaceException):
    
        @interface
        def function(a: int): return
        a = function(None) 
        
def test_unallowed_explicit_nonepassing_with_default():
    
    with pytest.raises(InterfaceException):    
        @interface
        def function(a: int, b: int=None): return
        a = function(None)


def test_wrong_default():
    
    with pytest.raises(InterfaceException):
        @interface
        def function(a: int=3.14): return
        a = function()
    