# standard library imports
import math
        
# third party imports
import numpy as np
        
# sub package imports
from core.static import interface

@interface
def feq(a: float|np.float16|np.float32|np.float64, b: float|np.float16|np.float32|np.float64, *, tol=1e-6):
    """
    returns true if a and b are within tol of difference
    """
    
    # works with infinite
    if (a == b): return True
    
    # compute diff
    d = math.fabs(a - b)
    
    return d < tol