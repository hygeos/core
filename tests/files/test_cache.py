from pathlib import Path
import pytest
from core.files import cache
from xarray.tutorial import open_dataset
from tempfile import TemporaryDirectory


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

@pytest.mark.parametrize('extension', ["pickle", "csv"])
def test_cache_dataframe(extension):
    def my_function():
        return open_dataset('air_temperature').to_dataframe().reset_index().round()
    with TemporaryDirectory() as tmpdir:
        cache_file = Path(tmpdir)/('cache.'+extension)
        cache.cache_dataframe(cache_file)(my_function)()
        a = cache.cache_dataframe(cache_file)(my_function)()
        assert all(a == my_function())

def test_cache_dataset():
    def my_function():
        return open_dataset('air_temperature')
    with TemporaryDirectory() as tmpdir:
        cache_file = Path(tmpdir)/'cache.nc'
        cache.cache_dataset(cache_file,
                      attrs={'a': 1}
                      )(my_function)()
        a = cache.cache_dataset(
            cache_file,
            chunks={'lat': 10, 'lon': 10},
        )(my_function)()
        assert a == my_function()
    