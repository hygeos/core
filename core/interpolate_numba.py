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

    Note:
        Uses an adaptive walk strategy — for each dimension, the search starts
        from the interval found for the previous query point, walking left or
        right as needed. This is O(1) amortized per point when successive query
        points are spatially close, which is typical in grid interpolation.
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
        # Use adaptive walk from previous index — O(1) amortized when successive
        # query points are close (common in grid interpolation workloads).
        index = 0  # running interval index for this dimension
        for j in range(n_points):
            value = xi[dim, j]

            # Handle NaN values
            if value != value:  # NaN check (numba compatible)
                indices[dim, j] = 0
                distances[dim, j] = np.nan
                continue

            # Clamp index to valid range (handles special -1 from length-1 grids)
            if index < 0:
                index = 0
            elif index >= coord_size - 1:
                index = coord_size - 2

            # Walk left if value falls before current interval
            while index > 0 and value < coordinates[index]:
                index -= 1

            # Walk right if value falls after current interval
            while index < coord_size - 2 and value >= coordinates[index + 1]:
                index += 1

            indices[dim, j] = index

            # Calculate normalized distance within interval
            left_val = coordinates[index]
            right_val = coordinates[index + 1]
            distances[dim, j] = (value - left_val) / (right_val - left_val)

    return (indices, distances)