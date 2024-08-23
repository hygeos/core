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
    blacklist = []
    
    # filters = ...
    def set_lvl(lvl: lvl):
        Log._min_global_level = lvl

    def format_msg(level, msg, mod):
        return f"{level.color}[{level.name}]: ({mod}.py){rgb.default} {msg}"
    
    def silence(*modules):  # TODO blacklist only below certain level ? e.g Log.silence("core.filegen", below=lvl.ERROR)
        for mod in modules:
            Log.blacklist.append(mod)
    
    def _log(lvl: lvl, msg: str, mod: str):
        
        for blacklisted_domain in Log.blacklist:
            if mod.startswith(blacklisted_domain):
                return
        
        if lvl.value >= Log._min_global_level.value: # and section not in LOG.filters:
           print(Log.format_msg(lvl, msg, mod))

# proxy
def log(lvl: lvl, msg: str):
    """
    log with selected lvl
    """
    
    # get calling module full name
    stk = inspect.stack()[1]
    mod = inspect.getmodule(stk[0]).__name__
    
    Log._log(lvl, msg, mod)

def info(msg: str):
    """
    log with default level INFO
    """
    
    log(lvl.INFO, msg)