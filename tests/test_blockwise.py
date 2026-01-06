

"""
Design of modules that apply to xarray Datasets, with the possibility of chaining
several modules
"""

from typing import List, Dict, Any

import numpy as np
import pytest
import xarray as xr

from core.process.blockwise import BlockProcessor, map_blockwise
from core.tools import Var


class InitProcessor(BlockProcessor):
    def input_vars(self) -> List[str]:
        return ['latitude']

    def created_vars(self) -> List[Var]:
        return [Var('flags', 'uint16', 'latitude')]

    def process_block(self, block: xr.Dataset, **kwargs):
        block['flags'] = xr.zeros_like(block.latitude, dtype='uint16')
        
class NDVIProcessor(BlockProcessor):
    """Example processor that computes NDVI from red and NIR bands."""
    
    def input_vars(self) -> List[str]:
        return ['latitude', 'rho_toa']  # Expecting multi-band reflectance data
    
    def created_vars(self) -> List[Var]:
        return [
            Var(
                "ndvi",
                "float32",
                "latitude",
                attrs={
                    "units": "dimensionless",
                    "long_name": "Normalized Difference Vegetation Index",
                },
            )
        ]
    
    def global_attrs(self) -> Dict[str, Any]:
        return {'processing': 'ndvi_computed', 'algorithm_version': '1.0'}
    
    def process_block(self, block: xr.Dataset, **kwargs) -> None:
        """Compute NDVI = (NIR - Red) / (NIR + Red)."""
        red_band = kwargs.get('red_band', 1)
        nir_band = kwargs.get('nir_band', 4)
        
        rho = block['rho_toa']
        red = rho.sel(band=red_band)
        nir = rho.sel(band=nir_band)
        
        # Compute NDVI with safe division
        numerator = nir - red
        denominator = nir + red
        ndvi = xr.where(denominator != 0, numerator / denominator, 0.0)
        
        # Add to block
        block['ndvi'] = ndvi.astype('float32')


class ThresholdProcessor(BlockProcessor):
    """Example processor that applies a threshold to create a mask."""
    
    def __init__(self, input_var: str = 'ndvi', threshold: float = 0.3):
        self._input_var = input_var
        self.threshold = threshold
    
    def input_vars(self) -> List[str]:
        return ['latitude', self._input_var]
    
    def modified_vars(self) -> List[Var]:
        return [Var('flags', flags={"ABOVE_THRESHOLD": 1})]

    def created_vars(self) -> List[Var]:
        return [
            Var(
                f"{self._input_var}_mask",
                "bool",
                "latitude",
                attrs={"long_name": f"{self._input_var} mask"},
            )
        ]

    def process_block(self, block: xr.Dataset, **kwargs) -> None:
        """Create a boolean mask based on threshold."""
        input_data = block[self._input_var]
        mask = input_data > self.threshold
        block[f'{self._input_var}_mask'] = mask
        
        self.raiseflag(block, "flags", "ABOVE_THRESHOLD", mask)


def create_sample_dataset(
    x_size: int = 500,
    y_size: int = 500,
    n_bands: int = 10,
    chunks: dict | None = {"x": 50, "y": 50, "band": -1},
    seed: int = 42,
) -> xr.Dataset:
    """
    Create a sample dataset with latitude, longitude, and rho_toa variables.
    
    Parameters
    ----------
    x_size, y_size : dimensions of the product
    n_bands : int
        Number of spectral bands (default: 4)
    chunks : dict, optional
        Chunk sizes for dask arrays. Example: {'lat': 45, 'lon': 90, 'band': 2}
    seed : int
        Random seed for reproducible data
        
    Returns
    -------
    xr.Dataset
        Dataset with latitude, longitude coordinates and rho_toa variable
    """
    np.random.seed(seed)

    # Create latitude and longitude 2D variables as random arrays
    latitude = np.random.uniform(-90, 90, size=(y_size, x_size))
    longitude = np.random.uniform(-180, 180, size=(y_size, x_size))

    ds = xr.Dataset(
        {
            "rho_toa": (
                ["band", "y", "x"],
                np.random.uniform(0.0, 1.0, size=(n_bands, y_size, x_size)),
                {
                    "long_name": "Top of Atmosphere Reflectance",
                    "units": "dimensionless",
                    "valid_range": [0.0, 1.0],
                },
            ),
            "latitude": (["y", "x"], latitude),
            "longitude": (["y", "x"], longitude),
        },
        attrs={
            "title": "Sample satellite reflectance dataset",
            "description": "Simulated top-of-atmosphere reflectance data",
            "created_by": "create_sample_dataset function",
        },
    )
    if chunks is not None:
        ds = ds.chunk(chunks)
    return ds


def test_sample_dataset():
    """Test the sample dataset creation."""
    # Create basic dataset
    ds = create_sample_dataset(x_size=50, y_size=50)

    # Check dimensions
    assert ds.sizes == {'x': 50, 'y': 50, 'band': 10}

    # Check variables
    assert 'rho_toa' in ds.data_vars
    assert 'latitude' in ds.data_vars
    assert 'longitude' in ds.data_vars

    # Check data ranges
    assert ds.rho_toa.min() >= 0.0
    assert ds.rho_toa.max() <= 1.0

    assert hasattr(ds.rho_toa.data, 'chunks')


def test_blockwise():
    ds = create_sample_dataset()
    
    # Create a chain of processors
    processors = [
        InitProcessor(),
        NDVIProcessor(),
        ThresholdProcessor('ndvi', 0.3)
    ]
    
    # Apply blockwise processing
    result = map_blockwise(ds, processors, red_band=1, nir_band=4)
    
    # Verify results
    assert 'ndvi' in result.data_vars
    assert 'ndvi_mask' in result.data_vars
    print("Blockwise processing test passed!")

    # Check that it computes
    result.compute()

    # Check attrs
    assert result['ndvi'].attrs == {'units': 'dimensionless', 'long_name': 'Normalized Difference Vegetation Index'}
    assert result['ndvi_mask'].attrs == {'long_name': 'ndvi mask'}
    
    # Check global attributes
    assert result.attrs['processing'] == 'ndvi_computed'
    assert result.attrs['algorithm_version'] == '1.0'
    # Original attributes should be preserved
    assert result.attrs['title'] == 'Sample satellite reflectance dataset'


def test_blockwise_newdims():
    class NewdimensionProcessor(BlockProcessor):

        def input_vars(self) -> List[str]:
            return ['latitude']
            
        def created_dims(self):
            return {'new_dim1': 2, 'new_dim2': [1, 2, 3]}

        def created_vars(self) -> List[Var]:
            return [
                Var("A", "float32", ("new_dim1", "new_dim2")),
                Var("B", "float32", ("new_dim1", "y", "x")),
            ]
        
        def process_block(self, block: xr.Dataset, **kwargs):
            block["A"] = xr.DataArray(
                np.zeros((2, 3), dtype="float32"), dims=("new_dim1", "new_dim2")
            )
            block["B"] = xr.DataArray(
                np.zeros((2, block.y.size, block.x.size), dtype="float32"),
                dims=("new_dim1", "y", "x"),
            )

    ds = create_sample_dataset()

    result = map_blockwise(ds, [NewdimensionProcessor()])
    assert 'new_dim1' in result.dims
    assert 'new_dim2' in result.dims
    assert 'new_dim2' in result.coords

    result.compute()

@pytest.mark.parametrize("mode", ["chained", "graphed"])
def test_perf(mode):
    """
    Check the performance of chained processing, compared with successive map_blocks

    There is a significant difference when napplication becomes larger than 10
    """
    napplication = 10
    ds = create_sample_dataset(x_size=10, y_size=10)

    class SampleProcessor(BlockProcessor):
        def modified_vars(self):
            return [Var('rho_toa')]
        def process_block(self, block: xr.Dataset, **kwargs):
            block['rho_toa'] += 1.

    from core.monitor import Chrono, dask_graph_stats

    if mode == "chained":
        result = map_blockwise(ds, [SampleProcessor()]*napplication)
    else:  # graphed
        result = ds.copy(deep=True)
        for _ in range(napplication):
            result = map_blockwise(result, [SampleProcessor()])

    print(dask_graph_stats(result).to_string(index=False))
    with Chrono(f'{mode.capitalize()}'):
        result['rho_toa'].mean().values

