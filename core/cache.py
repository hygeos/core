import warnings
from core.files.cache import (
    cache_dataframe,
    cache_json,
    cache_pickle,
    cache_dataset,
    cachefunc,
)

warnings.warn("Please import from core.files.cache", DeprecationWarning)
