"""
    Modular output module
    
    usage:
    
    from core.logging import log, lvl, Log
    or
    from core.logging import *
    
    log(lvl.LEVELNAME, message)
    
    Log.silence("module.path.to.be.silenced") 
    e.g
    Log.silence("core.static.interface")
    
"""
# standard library imports
from enum import Enum
import inspect
from tqdm import tqdm


# third party imports
# ...
        
# sub package imports
from core.output import rgb


# set levels enums
class lvl(Enum):
    DEBUG = 1
    INFO = 2 
    WARNING = 3
    ERROR = 4
    CRITICAL = 5

# set levels colors
lvl.DEBUG.color     = rgb.default
lvl.INFO.color      = rgb.default
lvl.WARNING.color   = rgb.orange
lvl.ERROR.color     = rgb.red
lvl.CRITICAL.color  = rgb.red

# Log storage class
class Log:
    _min_global_level = lvl.DEBUG
    
    #wrtiting to terminal by default
    _format_msg_func = None
    _loading_bar_func = tqdm
    
    blacklist = {}
    
    
    # filters = ...
    def set_lvl(lvl: lvl):
        Log._min_global_level = lvl

    def _format_msg(level: lvl, msg: str, mod):
        return f"{level.color}[{level.name}]: ({mod.__name__}.py){rgb.default} {msg}"
    
    def _format_msg_no_color(level: lvl, msg: str, mod):
        return f"[{level.name}]: ({mod.__name__}.py) {msg}"
       
    
    #here so that function are visible in order
    _format_msg_func = _format_msg
    
    
    def _no_loading_bar(iterable=None, desc=None, total=None, leave=True, file=None,
         ncols=None, mininterval=0.1, maxinterval=10.0, miniters=None,
         ascii=None, disable=False, unit='it', unit_scale=False,
         dynamic_ncols=False, smoothing=0.3, bar_format=None, initial=0,
         position=None, postfix=None, unit_divisor=1000, write_bytes=False,
         lock_args=None, nrows=None, colour=None, delay=0, gui=False,
         **kwargs):
        return iterable
    
    
    def set_to_file_output() :
        Log._format_msg_func = Log._format_msg_no_color
        Log._loading_bar_func = Log._no_loading_bar

    
    def set_to_terminal_output():
        Log._format_msg_func = Log._format_msg
        Log._loading_bar_func = tqdm
        
    


    
    #interface tqdm please go to https://tqdm.github.io/docs/tqdm/ for documentation
    def loading_bar(iterable=None, desc=None, total=None, leave=True, file=None,
         ncols=None, mininterval=0.1, maxinterval=10.0, miniters=None,
         ascii=None, disable=False, unit='it', unit_scale=False,
         dynamic_ncols=False, smoothing=0.3, bar_format=None, initial=0,
         position=None, postfix=None, unit_divisor=1000, write_bytes=False,
         lock_args=None, nrows=None, colour=None, delay=0, gui=False,
         **kwargs):
        
        
        return Log._loading_bar_func(iterable, desc, total, leave, file, ncols, mininterval,
                         maxinterval, miniters, ascii, disable, unit, unit_scale,
                         dynamic_ncols, smoothing, bar_format, initial, position,
                         postfix, unit_divisor, write_bytes, lock_args, nrows,
                         colour, delay, gui, **kwargs)
    
    
    def silence(module, bellow_level : lvl = lvl.CRITICAL):  # blacklist only below certain level ? e.g Log.silence("core.filegen", bellow_level=lvl.ERROR)
            Log.blacklist[module] = bellow_level
    
    def _log(lvl: lvl, msg: str, mod):
        
        if mod in Log.blacklist:
            if Log.blacklist[mod].value >= lvl.value :
                return
            
        

        if lvl.value >= Log._min_global_level.value: # and section not in LOG.filters:
           print(Log._format_msg_func(lvl, msg, mod))

# proxy
def log(lvl: lvl, msg: str):
    """
    log with selected lvl
    """
    
    # get calling module full name
    stk = inspect.stack()[1]
    mod = inspect.getmodule(stk[0])
    
    Log._log(lvl, msg, mod)

def info(msg: str):
    """
    log with default level INFO
    """
    
    log(lvl.INFO, msg)