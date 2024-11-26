import core
from core import log

import pytest

"""
only sensible in debug mode
"""

def test_base():
    
    log.silence(core, log.lvl.ERROR)
    
    log.warning("test string")