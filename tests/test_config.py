# standard library imports
from tempfile import TemporaryDirectory
from pathlib import Path
import os
        
# third party imports
import pytest
        
# sub package imports
from core import config

def test_config_usage():
    
    with TemporaryDirectory() as tmpdir:
        
        tmpdir = Path(tmpdir)
        
        cwd = Path.cwd()
        
        # go to tmpdir
        os.chdir(tmpdir)
        
        config.init("PATH/TO/DATA_DIR") # CLI usage
        
        config.load() # should be implicit at package import ?
    
        print(config.get("data", "description"))
        
        data_dir = config.get("data", "dir")
        anci_dir = config.get("data", "ancillary", default=Path(data_dir)/"ancillary")
        
        # restore working directory
        os.chdir(cwd)
        