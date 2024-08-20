from inspect import isclass, signature, _empty
from typing import get_origin, get_args

from core.static.Exceptions import WrongUsage, InterfaceException


def interface(function):
    """
    Declare a function or method as an Interface
    Raise an error if types passed do not match definition
    """
    
    if not interface.enabled:
        return function
    
    if isclass(function):
        raise WrongUsage(f'\n\tCannot declare class \'{function.__name__}\' as an interface, only functions or methods can be')
        
    def wrapper(*args, **kwargs):
        
        # construct datastructures used
        expected_signature = [(i.name, i.annotation) for i in signature(function).parameters.values()]
        unnamed_params = [type(i) for i in args] 
        named_params  = [(i, type(kwargs[i])) for i in kwargs] # named parameters can only be lasts 
        full_default_values = [(k, v.default) for k, v in signature(function).parameters.items()]
        default_params = [item for item in full_default_values if item[1] is not _empty]
        
        # DEBUG
        # print(expected_signature)
        # print(unnamed_params)
        # print(named_params)
        # print(default_params)
        
        # checknumber of parameters
        exp_nargs = len(expected_signature)
        act_nargs = len(unnamed_params) + len(named_params) + len(default_params)
        
        if exp_nargs > act_nargs:
            raise InterfaceException(f'\n\tFunction \'{function.__name__}\': Exepected {exp_nargs} arguments, got {act_nargs}')
        
        errors = []
        # check unnamed parameters
        for param_type in unnamed_params:
            expected_name, expected_type = expected_signature.pop(0)
            param_name, default_value = full_default_values.pop(0)
            
            if expected_type == _empty: continue
            
            if get_origin(expected_type) is type(int|float): # check if unions ( type(int|float) evaluate to typing.UnionType )
                expected_type = get_args(expected_type)
            
            if hasattr(expected_type, '__origin__'): # workaround for defs like list[str] → list (only check base type)
                expected_type = expected_type.__origin__
                
            if not issubclass(param_type, expected_type):
                explicit_none_passing = (param_type is type(None)) and (type(default_value) is type(None))
                if not explicit_none_passing: # no error if explicitly passed None, when None is the default, otherwise error
                    errors.append((expected_name, expected_type, param_type))
        
        
        expected_signature  = {i[0]: i[1] for i in expected_signature}
        full_default_values = {i[0]: i[1] for i in full_default_values}
        
        # check named parameters
        for param_name, param_type in named_params:
            expected_type = expected_signature.pop(param_name)
            default_value = full_default_values.pop(param_name) # only check in case of error: exempt None if default
            
            if expected_type == _empty: continue
            
            if get_origin(expected_type) is type(int|float): # allow unions
                expected_type = get_args(expected_type)
            
            if hasattr(expected_type, '__origin__'): # workaround for defs like list[str] → list (only check base type)
                expected_type = expected_type.__origin__
                
            if not issubclass(param_type, expected_type):
                explicit_none_passing = (param_type is type(None)) and (type(default_value) is type(None))
                if not explicit_none_passing: # no error if explicitly passed None, when None is the default, otherwise error
                    errors.append((param_name, expected_type, param_type))
        
        for param_name, param_value in full_default_values.items():
            param_type = type(param_value)
            
            expected_type = expected_signature.pop(param_name)
            
            if (param_type is type(None)): continue # no need to specify that None is a possible value if it is the default value
            
            if not issubclass(param_type, expected_type):
                errors.append((param_name, expected_type, param_type))
    
        # raise error if at least one mismatch
        if len(errors) != 0: # error on at least one parameter
            mess = f'\n\tFunction \'{function.__name__}\': Wrong parameters types passed:'  
            for p in errors:
                param, expect, actual = p
                if type(p[1]) is tuple: # better message for unions e.g: int | float
                    expect = " or ".join([str(i) for i in p[1]])
                
                mess += (f'\n\t\tParameter \'{param}\' expected: {expect} got {actual}')
            raise InterfaceException(mess)

        # check if some arguments haven't been checked        
        if len(expected_signature) != 0:
            mess = [f"Parameter \'{p}\' still unchecked after interface call, module error." for p in expected_signature]
            raise InterfaceException(mess)
            
        return function(*args, **kwargs)
    return wrapper


interface.enabled = True