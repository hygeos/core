# standard library imports
from tempfile import TemporaryDirectory
from pathlib import Path
import shutil
import os
        
# third party imports
import pytest
        
# sub package imports
from core import config

def test_config_usage():
    
    with TemporaryDirectory() as tmpdir:
        
        # go to tmpdir
        os.chdir(tmpdir)
        config.init(Path.cwd()) # supposed to be CLI usage
        
        data_dir = config.get("general", "base_dir", default="/archive2/data", write=True) # default=str(tmpdir), write=True)
        anci_dir = config.get("harp", "data_dir", default=Path(data_dir)/"ancillary")
        


def test_config_base():
    
    # necessary for tests
    config._load_for_test_only("tests/config/core-config.toml")
    
    a = config.get("harp", "anci_dir")
    assert a == "path/to/somewhere/else"
    
    b = config.get("general", "base_dir")
    assert b == "test/path/data"

def test_config_missing():

    config._load_for_test_only("tests/config/core-config.toml")

    with pytest.raises(KeyError):
        c = config.get("Missing_section", "key")

    with pytest.raises(KeyError):
        d = config.get("general", "missing_key")

def test_config_default():

    config._load_for_test_only("tests/config/core-config.toml")

    with pytest.raises(KeyError):
        c = config.get("Missing section", "key", default= -1)
        assert c == -1

def test_config_default_writing():

    source = Path("tests/config/core-config.toml")
    dest = Path("tests/config/core-config-copy.toml")

    shutil.copy(source, dest)

    config._load_for_test_only(dest) # not needed outside of tests

    with pytest.raises(KeyError):
        c = config.get("general", "missing_key")

    # Serialization and deserialisation test
    d = config.get("general", "missing_key", default=123, write=True)
    e = config.get("general", "missing_key") 
    assert e == 123
    
    f = config.get("general", "missing_key_2", default=3.14, write=True)
    g = config.get("general", "missing_key_2") 
    assert g == 3.14
    
    
    h = config.get("general", "missing_key_3", default="string test", write=True)
    i = config.get("general", "missing_key_3") 
    assert i == "string test"
    
    dest.unlink() # remove file
        