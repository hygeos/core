"""
    Module Shadowed by API function of same name log
    
    usage:
    
    from core import log
    
"""
# standard library imports
from datetime import datetime
from enum import Enum
import inspect

# third party imports
# ...
        
# sub package imports
# ...


class config:
    show_level = True
    show_namespace = False
    show_time = True
    show_color = True


class _color:
    silenced = False
    
    def __init__(self, value):
        self.string = value

    def __str__(self):
        return self.string if config.show_color else ""


class rgb:
    purple      = _color('\033[95m')
    blue        = _color('\033[94m')
    cyan        = _color('\033[96m')
    green       = _color('\033[92m')
    orange      = _color('\033[93m')
    red         = _color('\033[91m')
    bold        = _color('\033[1m')
    underline   = _color('\033[4m')
    default     = _color('\033[0m')
    
# set levels enums
class lvl(Enum):
    DEBUG = 1
    INFO = 2 
    WARNING = 3
    ERROR = 4
    CRITICAL = 5

# set levels colors
lvl.DEBUG.color     = rgb.purple
lvl.INFO.color      = rgb.blue
lvl.WARNING.color   = rgb.orange
lvl.ERROR.color     = rgb.red
lvl.CRITICAL.color  = rgb.red
    
class _internal:
    min_global_level = lvl.DEBUG
    blacklist = {}
    
    def format_msg(level: lvl, msg: str, mod):
        
        lvl_prefix = "" # construct prefixes depending on options
        namespace_prefix = ""
        time_prefix = ""
        
        if config.show_level:       lvl_prefix          += f"[{level.name}] "
        if config.show_namespace:   namespace_prefix    += f"({mod.__name__}) "
        if config.show_time:        time_prefix         += f"{datetime.now().strftime("%H:%M:%S")} "
        
        string = f"{level.color}{lvl_prefix}{rgb.orange}{namespace_prefix}{rgb.green}{time_prefix}{rgb.default}{msg}"
        
        return string
        
        # proxy
    def log(lvl: lvl, *args):
        """
        log with selected lvl
        """
        
        msg = _internal.concat_mess(*args)
        
        # get calling module full name
        stk = inspect.stack()[1]
        mod = inspect.getmodule(stk[0])
        
        if lvl.value < _internal.min_global_level.value: # and section not in LOG.filters:
            return
        
        for blmod in _internal.blacklist: # blacklisted module
            blname = blmod.__name__
            if blname in mod.__name__ and lvl.value <= _internal.blacklist[blmod].value:
                return 
                
        print(_internal.format_msg(lvl, msg, mod))
    
    def concat_mess(*args):
        message = ""
        for arg in args:
            message += str(arg)
        return message + str(rgb.default)
        
# filters = ...
def set_lvl(lvl: lvl):
    _internal.min_global_level = lvl

# def _no_loading_bar(iterable=None, desc=None, total=None, leave=True, file=None,
#         ncols=None, mininterval=0.1, maxinterval=10.0, miniters=None,
#         ascii=None, disable=False, unit='it', unit_scale=False,
#         dynamic_ncols=False, smoothing=0.3, bar_format=None, initial=0,
#         position=None, postfix=None, unit_divisor=1000, write_bytes=False,
#         lock_args=None, nrows=None, colour=None, delay=0, gui=False,
#         **kwargs):
#     return iterable


# def set_to_file_output() :
#     Log._format_msg_func = Log._format_msg_no_color
#     Log._loading_bar_func = Log._no_loading_bar


# def set_to_terminal_output():
#     Log._format_msg_func = Log._format_msg
#     Log._loading_bar_func = tqdm
    
# #interface tqdm please go to https://tqdm.github.io/docs/tqdm/ for documentation
# def loading_bar(iterable=None, desc=None, total=None, leave=True, file=None,
#         ncols=None, mininterval=0.1, maxinterval=10.0, miniters=None,
#         ascii=None, disable=False, unit='it', unit_scale=False,
#         dynamic_ncols=False, smoothing=0.3, bar_format=None, initial=0,
#         position=None, postfix=None, unit_divisor=1000, write_bytes=False,
#         lock_args=None, nrows=None, colour=None, delay=0, gui=False,
#         **kwargs):
    
    
#     return Log._loading_bar_func(iterable, desc, total, leave, file, ncols, mininterval,
#                         maxinterval, miniters, ascii, disable, unit, unit_scale,
#                         dynamic_ncols, smoothing, bar_format, initial, position,
#                         postfix, unit_divisor, write_bytes, lock_args, nrows,
#                         colour, delay, gui, **kwargs)


def silence(module, level_and_below : lvl = lvl.CRITICAL):  # blacklist only below certain level ? e.g Log.silence("core.filegen", bellow_level=lvl.ERROR)
        _internal.blacklist[module] = level_and_below


def __call__(lvl: lvl, *args):
    """
    log with specified level
    """
    _internal.log(lvl, *args)

def critical(*args):
    """
    log with defaul level CRITICAL
    """
    _internal.log(lvl.CRITICAL, *args)
    
def error(*args):
    """
    log with defaul level ERROR
    """
    _internal.log(lvl.ERROR, *args)
    
def warning(*args):
    """
    log with defaul level WARNING
    """
    _internal.log(lvl.WARNING, *args)
    
def info(*args):
    """
    log with default level INFO
    """
    _internal.log(lvl.INFO, *args)
    
def debug(*args):
    """
    log with defaul level DEBUG
    """
    _internal.log(lvl.DEBUG, *args)
