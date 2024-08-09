# standard library imports
import math
        
# third party imports
# ...
        
# sub package imports
# ...

def feq(a: float, b: float, *, tol=1e-6):
    """
    returns true if a and b are within tol of difference
    """
    
    # works with infinite
    if (a == b): return True
    
    # compute diff
    d = math.fabs(a - b)
    
    return d < tol