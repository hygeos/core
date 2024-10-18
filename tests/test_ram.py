# standard library imports
import numpy as np
        
# third party imports
import pytest
        
# sub package imports
from core import RAM


def test_base():
    
    def big_function():
        a = [np.random.randint(0,10,(20,20)) for i in range(20)]
        return len(a)
    
    with RAM("Computation N°1"):
        res = big_function()
        print("result", res)
    
    print('end')