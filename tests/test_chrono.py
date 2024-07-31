import pytest

from core import Chrono

def test_base():
    
    def big_function():
        mysum = 0
        for i in range(999999):
            if i % 3 == 0:
                mysum += 1
                
        return mysum
    
    with Chrono("Computation N°1"):
        res = big_function()
        print("result", res)
    
    print('end')
    