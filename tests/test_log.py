import core
from core import log
from time import sleep

import pytest

"""
only sensible in debug mode
"""

def test_base():
    
    log.silence(core, log.lvl.ERROR)
    
    log.warning("test string")

def test_str_conversion():
    log.log(log.lvl.INFO, 1)
    
    
def test_box():
    
    log.info(
        "Let's put emphasis on ",
        log.rgb.orange("THIS"),
        " word"
    )

def test_progress_bar():
    pbar = log.pbar(log.lvl.INFO, range(20), desc='test')
    for i in pbar:
        pbar.write(str(i))
        sleep(0.1)