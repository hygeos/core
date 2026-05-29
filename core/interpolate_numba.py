"""
Numba-accelerated functions for interpolation.

Provides a JIT-compiled version of find_indices for multi-dimensional
grid interpolation preprocessing.
"""

from typing import Tuple

import numpy as np
from numba import jit, prange


@jit(nopython=True, parallel=True)
def find_indices_numba(grid, xi) -> Tuple[np.ndarray, np.ndarray]:
    """
    Multi-dimensional grid interpolation preprocessing.

    Args:
        grid: Tuple of 1D arrays defining grid coordinates for each dimension
        xi: 2D array where each row represents a dimension and each column a query point

    Returns:
        indices: Grid interval indices for each query point in each dimension
        distances: Normalized distances within each interval
    """
    n_dims, n_points = xi.shape

    # Use np.intp for indices and np.float32 for distances
    indices = np.empty((n_dims, n_points), dtype=np.intp)
    distances = np.empty((n_dims, n_points), dtype=np.float32)

    # NOTE: changing the parallelization strategy could improve performance
    for dim in prange(n_dims):  # parallel over dimensions
        coordinates = grid[dim]
        coord_size = len(coordinates)

        # Handle length-1 grid special case
        if coord_size == 1:
            for j in range(n_points):
                indices[dim, j] = -1  # Special hack value for downstream processing
                distances[dim, j] = 0.0
            continue  # exit early

        # Process each point in this dimension
        for j in range(n_points):
            value = xi[dim, j]

            # Handle NaN values
            if value != value:  # NaN check (numba compatible)
                indices[dim, j] = 0  # Safe fallback index
                distances[dim, j] = np.nan
                continue

            # Find interval using searchsorted
            index = np.searchsorted(coordinates, value, side="left") - 1

            # Handle extrapolation by bringing index back to valid bounds
            if index < 0:
                index = 0
            elif index >= coord_size - 1:
                index = coord_size - 2

            indices[dim, j] = index

            # Calculate normalized distance within interval
            left_val = coordinates[index]
            right_val = coordinates[index + 1]
            distances[dim, j] = (value - left_val) / (right_val - left_val)

    return (indices, distances)