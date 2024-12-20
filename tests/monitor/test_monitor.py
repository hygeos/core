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