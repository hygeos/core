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
    
    
class TestHashParams:
    """Tests for the hashparams function."""
    
    def test_same_args_produce_same_hash(self):
        """Same arguments should always produce the same hash."""
        h1 = cache.hashparams(a=1, b=2, c=3, x=10, y=20)
        h2 = cache.hashparams(a=1, b=2, c=3, x=10, y=20)
        assert h1 == h2
    
    def test_different_args_produce_different_hash(self):
        """Different arguments should produce different hashes."""
        h1 = cache.hashparams(a=1, b=2, c=3)
        h2 = cache.hashparams(a=1, b=2, c=4)
        assert h1 != h2
    
    def test_kwargs_order_independent(self):
        """Hash should be independent of kwargs order."""
        h1 = cache.hashparams(x=1, y=2, z=3)
        h2 = cache.hashparams(z=3, x=1, y=2)
        h3 = cache.hashparams(y=2, z=3, x=1)
        assert h1 == h2 == h3
    
    def test_only_kwargs(self):
        """Test hashing with only keyword arguments."""
        h1 = cache.hashparams(a=1, b="test")
        h2 = cache.hashparams(a=1, b="test")
        assert h1 == h2
    
    def test_empty_args(self):
        """Test hashing with no arguments."""
        h1 = cache.hashparams()
        h2 = cache.hashparams()
        assert h1 == h2
        assert len(h1) == 32  # BLAKE2b with digest_size=16 produces 32 hex chars
    
    def test_different_types(self):
        """Test hashing various data types."""
        h1 = cache.hashparams(
            int_val=42,
            float_val=3.14,
            str_val="hello",
            list_val=[1, 2, 3],
            dict_val={"a": 1},
            tuple_val=(1, 2),
            none_val=None,
            bool_val=True
        )
        h2 = cache.hashparams(
            int_val=42,
            float_val=3.14,
            str_val="hello",
            list_val=[1, 2, 3],
            dict_val={"a": 1},
            tuple_val=(1, 2),
            none_val=None,
            bool_val=True
        )
        assert h1 == h2
    
    def test_hash_format(self):
        """Test that hash is a valid hexadecimal string of correct length."""
        h = cache.hashparams(a=1, b=2, c=3, x=10)
        assert len(h) == 32  # 16 bytes = 32 hex characters
        assert all(c in '0123456789abcdef' for c in h)  # valid hex
    
    def test_positional_args_rejected(self):
        """Positional args should raise TypeError."""
        with pytest.raises(TypeError):
            cache.hashparams(1, 2, 3)
        
        with pytest.raises(TypeError):
            cache.hashparams(1, x=1)
    
    def test_nested_structures(self):
        """Test hashing nested data structures."""
        nested = {
            "level1": {
                "level2": [1, 2, {"level3": "deep"}]
            }
        }
        h1 = cache.hashparams(data=nested)
        h2 = cache.hashparams(data=nested)
        assert h1 == h2