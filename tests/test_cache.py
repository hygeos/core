from tempfile import TemporaryDirectory
from pathlib import Path
import pytest

from core import cache


@pytest.mark.parametrize('cache_function,var', [
    (cache.cache_json, 1),
    (cache.cache_json, 'a'),
    (cache.cache_json, [1, 'b', [1, 2]]),
    (cache.cache_pickle, [1, 'b', [1, 2]]),
])
def test_cachefunc(cache_function, var):
    def my_function(*args):
        return var
        
    with TemporaryDirectory() as tmpdir:
        cache_file = Path(tmpdir)/'cache.json'

        cache_function(cache_file)(my_function)()
        a = cache_function(cache_file)(my_function)()
        
        with pytest.raises(ValueError):
            # raise an error when called with the wrong arguments
            cache_function(cache_file)(my_function)(0)

        assert a == my_function()
        
def test_cache_json():
            
    def square_function(n):
        return n*n
    
    with TemporaryDirectory() as tmpdir:
        cache_file = Path(tmpdir)/'cache.json'
        
        param = 3
    
        a = cache.cache_json(cache_file)(square_function)(param)
        # second call
        b = cache.cache_json(cache_file)(square_function)(param)
        
        assert a == square_function(param) == b