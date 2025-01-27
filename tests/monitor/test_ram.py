# standard library imports
import numpy as np
        
# third party imports
import pytest
        
# sub package imports
from core.monitor import RAM


def test_base():
    
    def big_function():
        a = [np.random.randint(0,10,(20,20)) for i in range(20)]
        return len(a)
    
    with RAM("Computation N°1"):
        res = big_function()
        print("result", res)
    
    print('end')

def test_nonpaused():
    
    r = RAM()
    dt = r.elapsed()
    a = [np.random.randint(0,10,(20,20)) for i in range(1000)]
    dt2 = r.elapsed()
    
    assert not dt2 == dt

def test_pause():
    r = RAM()
    r.pause()
    dt = r.elapsed()
    a = [np.random.randint(0,10,(20,20)) for i in range(20)]
    dt2 = r.elapsed()
    
    assert dt == dt2

def test_loop():
    a,b = [], []
    r = RAM()
    for i in range(10):
        r.restart()
        a.append(np.random.randint(0,10,(20,20)))
        r.pause()
        b.append(np.random.randint(0,10,(20,20)))
    pass
    # assert pytest.approx(c.total_t, 0.01) == 10*0.01