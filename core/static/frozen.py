from inspect import isclass

from .Exceptions import *


def frozen(my_class):
    """
    Declare a class as frozen after construction,
    Raise an error if attempting to add attributes post constructor to an instance
    Doesn't apply to Class attributes (and cannot be applied with a decorator)
    """
    
    if not isclass(my_class):
        raise WrongUsage(f'Can only apply @freeze to a class, and \'{my_class.__name__}\' is not a class')

    def _disable_new_attributes(self, key, value):
        
        if not hasattr(self, key) and hasattr(self, '_frozen'):
            raise ClassIsFrozen( f"Custom class '{type(self).__name__}' is a frozen class, cannot reassign new attributes")
        else:
            if key == "_frozen" and value is not True:
                raise ClassIsFrozen(f"Custom class '{type(self).__name__}' is a frozen class, you cannot manually override this attribute at runtime.")
                
            object.__setattr__(self, key, value)
    
    my_class.__setattr__ = _disable_new_attributes
    
    def _init_wrapper(init_func):
        def inner(self, *args, **kwargs):
            init_func(self, *args, **kwargs)
            self._frozen = True # freeze object after its constructor
        return inner
    
    my_class.__init__ = _init_wrapper(my_class.__init__)
    
    return my_class