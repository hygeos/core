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
        
        config.init(Path.cwd()) # CLI usage
        
        data_dir = config.get("general", "base_dir", default="/archive2/data", write=True) # default=str(tmpdir), write=True)
        anci_dir = config.get("harp", "data_dir", default=Path(data_dir)/"ancillary")
        

        # restore working directory
        os.chdir(cwd)
        