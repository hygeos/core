"""
    Module Shadowed by API function of same name log
    
    usage:
    
    from core import log
    
"""
# standard library imports
from datetime import datetime
from enum import Enum
from tqdm import tqdm
import inspect
import warnings

# third party imports
import sys
if 'ipykernel' in sys.modules: from tqdm.notebook import tqdm
else: from tqdm import tqdm
        
# sub package imports
# ...


class config:
    show_level = True
    show_namespace = False
    show_time = True
    show_color = True


class _color:
    default = '\033[0m'
    silenced = False
    
    def __init__(self, value):
        self.string = value

    def __str__(self):
        return self.string if config.show_color else ""
    
    def __call__(self, string):
        """
        boxes the provided string with its color, and reset to default afterward
        """
        return self.string + str(string) + _color.default

class rgb:
    purple      = _color('\033[95m')
    blue        = _color('\033[94m')
    cyan        = _color('\033[96m')
    green       = _color('\033[92m')
    orange      = _color('\033[93m')
    red         = _color('\033[91m')
    bold        = _color('\033[1m')
    underline   = _color('\033[4m')
    default     = _color(_color.default)
    
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
        
        mod_name = "main" if not hasattr(mod, "__name__") else mod.__name__ # if calling from main mod is None
        
        if config.show_level:       lvl_prefix          += f"[{level.name}] ".ljust(8+2)
        if config.show_namespace:   namespace_prefix    += f"({mod_name}) "
        if config.show_time:        time_prefix         += f"{datetime.now().strftime('%H:%M:%S')} "
        
        string = f"{level.color}{lvl_prefix}{rgb.orange}{namespace_prefix}{rgb.green}{time_prefix}{rgb.default}{msg}"
        
        return string
    
        
        # proxy
    def log(lvl: lvl, *args):
        """
        log with selected lvl
        """
        
        # get calling module full name
        frame  = inspect.currentframe().f_back.f_back
        mod    = inspect.getmodule(frame)
        
        msg = _internal.concat_mess(*args)
        
        if lvl.value < _internal.min_global_level.value: # and section not in LOG.filters:
            return
        
        for blmod in _internal.blacklist: # blacklisted module
            blname = blmod.__name__
            if blname in mod.__name__ and lvl.value <= _internal.blacklist[blmod].value:
                return 
                
        print(_internal.format_msg(lvl, msg, mod), file=sys.stderr)
    
    def _loading_bar(**kwargs):
        frame  = inspect.currentframe().f_back.f_back
        mod = inspect.getmodule(frame)
        
        lvl = kwargs.pop('lvl')
        prefix = _internal.format_msg(lvl, '', mod)
        # kwargs['bar_format'] = prefix+'{l_bar}{bar}{r_bar}'
        kwargs['bar_format'] = '{l_bar}{bar}{r_bar}'
        kwargs.pop('kwargs')
        return tqdm(**kwargs)
    
    def concat_mess(*args):
        message = ""
        for arg in args:
            message += str(arg)
        return message + str(rgb.default)
        
def stderr(*args):
    msg = _internal.concat_mess(*args)
    print(msg, file=sys.stderr)
        
# filters = ...
def set_lvl(lvl: lvl):
    _internal.min_global_level = lvl

    
#interface tqdm please go to https://tqdm.github.io/docs/tqdm/ for documentation
def pbar(lvl: lvl = lvl.INFO, iterable=None, desc=None, total=None, leave=True, 
        ncols=None, mininterval=0.1, maxinterval=10.0, miniters=None,
        ascii=None, disable=False, unit='it', unit_scale=False,
        dynamic_ncols=False, smoothing=0.3, initial=0, file=None,
        position=None, postfix=None, unit_divisor=1000, write_bytes=False,
        lock_args=None, nrows=None, colour=None, delay=0, gui=False,
        **kwargs):
    """
    log a progress bar such as tqdm
    Args: All arguments are those from tqdm (cf. https://tqdm.github.io/docs/tqdm/)
    """
    
    return _internal._loading_bar(**locals())


def silence(module, level_and_below : lvl = lvl.CRITICAL):  # blacklist only below certain level ? e.g Log.silence("core.filegen", bellow_level=lvl.ERROR)
        _internal.blacklist[module] = level_and_below


def log(lvl: lvl, *args):
    """
    log with specified level
    """
    
    _internal.log(lvl, *args)

def critical(*args, e: Exception=RuntimeError):
    """
    log with defaul level CRITICAL
    will raise e if passed
    """
    
    _internal.log(lvl.CRITICAL, *args)
    
    if e is not None:
        if not issubclass(e, Exception):
            raise RuntimeError(f"log.error Invalid Exception type: {str(e)}, should be a subclass of {str(Exception)}")
        raise e(*args)
    
def error(*args, e: Exception=RuntimeError):
    """
    log with defaul level ERROR
    will raise e if passed
    """

    _internal.log(lvl.ERROR, *args)
    
    if e is not None:
        if not issubclass(e, Exception):
            raise RuntimeError(f"log.error Invalid Exception type: {str(e)}, should be a subclass of {str(Exception)}")
        raise e(*args)
    
def warning(*args, w: Warning=None):
    """
    log with defaul level WARNING
    """
    _internal.log(lvl.WARNING, *args)
    
    if w is not None:
        if not issubclass(w, Warning):
            raise RuntimeError(f"log.error Invalid Warning type: {str(e)}, should be a subclass of {str(Warning)}")
            
        msg = _internal.concat_mess(*args)
        warnings.warn(msg, category=w)
    
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



