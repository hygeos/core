# standard library imports
import numpy as np
import time

# third party imports
import pytest
        
# sub package imports
from core.monitor import Monitor


def test_base():
    
    def big_function():
        a = [np.random.randint(0,10,(20,20)) for i in range(20)]
        time.sleep(1)
        return len(a)
    
    with Monitor():
        res = big_function()
        print("result", res)
    
    print('end')

def test_nonpaused():
    
    m = Monitor()
    out1 = m.elapsed()
    time.sleep(0.01)
    a = [np.random.randint(0,10,(20,20)) for i in range(20)]
    out2 = m.elapsed()
    
    assert all(dt1 != dt2 for dt1, dt2 in zip(out1, out2))
    

def test_pause():
    m = Monitor()
    m.pause()
    out1 = m.elapsed()
    time.sleep(0.01)
    a = [np.random.randint(0,10,(20,20)) for i in range(20)]
    out2 = m.elapsed()
    
    assert all(dt1 == dt2 for dt1, dt2 in zip(out1, out2))

def test_loop():
    m = Monitor()
    for i in range(10):
        m.restart()
        time.sleep(0.01)
        m.pause()
        time.sleep(0.01)
    m.display()