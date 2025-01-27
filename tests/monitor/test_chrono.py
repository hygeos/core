# standard library imports
import time
        
# third party imports
import pytest
        
# sub package imports
from core.monitor import Chrono


def test_base():
    
    def big_function():
        time.sleep(0.01)
        return 42
    
    with Chrono("Computation N°1", unit="ms"):
        res = big_function()
        print("result", res)
    
    print('end')

def test_nonpaused():
    
    c = Chrono()
    dt = c.elapsed()
    time.sleep(0.01)
    dt2 = c.elapsed()
    
    assert not dt == dt2
    

def test_pause():
    c = Chrono()
    c.pause()
    dt = c.elapsed()
    time.sleep(0.01)
    dt2 = c.elapsed()
    
    assert dt == dt2

def test_loop():
    c = Chrono()
    for i in range(10):
        c.restart()
        time.sleep(0.01)
        c.pause()
        time.sleep(0.01)
    
    assert pytest.approx(c.total_t, rel=1e-1,) == 10*0.01