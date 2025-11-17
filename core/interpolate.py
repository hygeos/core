from datetime import datetime
from functools import reduce
from itertools import chain, product
from typing import Dict, Iterable, List, Literal, Callable

import numpy as np
import xarray as xr
from numpy.typing import NDArray

# from scipy.interpolate._rgi_cython import find_indices as scipy_find_indices

try:
    from numba import njit, jit, prange
    from numba.typed import List
    
except ImportError:
    # Numba is not available
    # -> Replace all import by native Python alternive
    
    List = list
    def njit(func: Callable) -> Callable: # no-op decorator
        return func
    def jit(func: Callable) -> Callable: # no-op decorator
        return func
    prange = range



def interp(da: xr.DataArray, **kwargs):
    """
    Interpolate/select a DataArray onto new coordinates.

    This function is similar to xr.interp and xr.sel, but:
        - Supports dask-based coordinates inputs without triggering immediate
          computation as is done by xr.interp
        - Supports combinations of selection and interpolation. This is faster and more
          memory efficient than performing independently the selection and interpolation.
        - Supports pointwise indexing/interpolation using dask arrays
          (see https://docs.xarray.dev/en/latest/user-guide/indexing.html#more-advanced-indexing)
        - Supports per-dimension options (nearest neighbour selection, linear/spline
          interpolation, out-of-bounds behaviour, cyclic dimensions...)

    Args:
        da (xr.DataArray): The input DataArray
        **kwargs: definition of the selection/interpolation coordinates for each
            dimension, using the following classes:
                - Linear: linear interpolation (like xr.DataArray.interp)
                - Nearest: nearest neighbour selection (like xr.DataArray.sel)
                - Index: integer index selection (like xr.DataArray.isel)
            These classes store the coordinate data in their `.values` attribute and have
            a `.get_indexer` method which returns an indexer for the passed coordinates.

    Example:
        >>> interp(
        ...     data,  # input DataArray with dimensions (a, b, c)
        ...     a = Linear(           # perform linear interpolation along dimension `a`
        ...          a_values,        # `a_values` is a DataArray with dimension (x, y);
        ...          bounds='clip'),  # clip out-of-bounds values to the axis min/max.
        ...     b = Nearest(b_values), # perform nearest neighbour selection along
        ...                            # dimension `b`; `b_values` is a DataArray
        ...                            # with dimension (x, y)
        ... ) # returns a DataArray with dimensions (x, y, c)
        No interpolation or selection is performed along dimension `c` thus it is
        left as-is.

    Returns:
        xr.DataArray: DataArray on the new coordinates.
    """
    assert (da.chunks is None) or (
        len(da.chunks) == 0
    ), "Input DataArray should not be dask-based"

    # merge all indexing DataArrays in a single Dataset
    ds = xr.Dataset({k: v.values for k, v in kwargs.items()})

    # get interpolators along all dimensions
    indexers = {k: v.get_indexer(da[k]) for k, v in kwargs.items()}

    # prevent common dimensions between da and the pointwise indexing dimensions
    assert not set(ds.dims).intersection(da.dims)
    
    # get unique dimensions from all variables in the dataset (preserving order)
    # we use this because ds.dims may change the order of dimensions
    ds_dims = list(dict.fromkeys(chain(*[ds[var].dims for var in ds])))

    # transpose ds to get fixed dimension ordering
    ds = ds.transpose(*ds_dims)
    
    out_dims = determine_output_dimensions(da, ds_dims, kwargs.keys())

    ret = xr.map_blocks(
        interp_block,
        ds,
        kwargs={
            "da": da,
            "out_dims": out_dims,
            "indexers": indexers,
        },
    )
    ret.attrs.update(da.attrs)

    return ret


def product_dict(**kwargs) -> Iterable[Dict]:
    """
    Cartesian product of a dictionary of lists
    """
    # https://stackoverflow.com/questions/5228158/
    keys = kwargs.keys()
    for instance in product(*kwargs.values()):
        yield dict(zip(keys, instance))


def interp_block(
    ds: xr.Dataset, da: xr.DataArray, out_dims, indexers: Dict
) -> xr.DataArray:
    """
    This function is called by map_blocks in function `interp`, and performs the
    indexing and interpolation at the numpy level.

    It relies on the indexers to perform index searching and weight calculation, and
    performs a linear combination of the sub-arrays.
    """
    # get broadcasted data from ds (all with the same number of dimensions)
    np_indexers = broadcast_numpy(ds)
    
    # get the shapes for broadcasting each weight against the output dimensions
    w_shape = broadcast_shapes(ds, out_dims)

    # apply index searching over all dimensions (ie, v(values))
    t0 = datetime.now()
    indices_weights = {k: v(np_indexers[k]) for k, v in indexers.items()}
    time_find_index = datetime.now() - t0

    # cartesian product of the combination of lower and upper indices (in case of
    # linear interpolation) for each dimension
    t0 = datetime.now()
    result = 0
    data_values = da.values
    for iw in product_dict(**indices_weights):
        weights = [
            1 if iw[dim][1] is None else iw[dim][1].reshape(w_shape[dim])
            for dim in iw
        ]
        w = reduce(lambda x, y: x*y, weights)
        keys = [(iw[dim][0] if dim in iw else slice(None)) for dim in da.dims]
        result += data_values[tuple(keys)] * w
    time_interp = datetime.now() - t0
    
    # determine output coords
    coords = {}
    for dim in out_dims:
        if dim in da.coords:
            coords[dim] = da.coords[dim]
        elif dim in ds.coords:
            coords[dim] = ds.coords[dim]

    # create output DataArray
    ret = xr.DataArray(
        result,
        dims=out_dims,
        coords=coords,
    )

    ret.attrs.update(
        {
            "time_find_index": time_find_index,
            "time_interp": time_interp,
        }
    )

    return ret


@jit(nopython=True, parallel=True)
def find_indices(grid, xi) -> tuple[np.ndarray, np.ndarray]:
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
    
    # Use np.intp for indices and np.float64 for distances
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
            continue # exit ealy
            
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

class Locator:
    """
    The purpose of these classes is to locate values in coordinate axes.
    """
    def __init__(self, coords: NDArray, bounds: str):
        self.coords = coords.astype("double")
        self.bounds = bounds

        if coords[0] < coords[-1]:
            self.vmin = coords[0]
            self.vmax = coords[-1]
        else:
            self.vmin = coords[-1]
            self.vmax = coords[0]

        if bounds == "cycle":
            raise RuntimeError('Cannot use bounds="cycle" with irregular locator')


    def handle_oob(self, values: np.ndarray):
        """
        handle out of bound values

        Note: when bounds == "cycle", does nothing
        """
        if self.bounds == "clip":
            # FIXME: manage the case where coords and values
            # have incompatible types (eg, int and float)
            values = values.clip(self.vmin, self.vmax)
        elif self.bounds in ["nan", "error"]:
            # bounds is either "nan" or "error"
            oob = (values < self.vmin) | (values > self.vmax)
            if oob.any():
                if self.bounds == "error":
                    raise ValueError
                if self.bounds == "nan":
                    values = np.where(oob, np.nan, values)

        return values
        
    
    def locate_index_weight(self, values):
        """
        Find indices and dist of values for linear and spline interpolation in self.coords

        Returns a list of indices, dist (float 0 to 1) and oob
        """
        values = self.handle_oob(values)

        shp = values.shape
        indices, dist = find_indices(
            (self.coords,), values.astype("double").ravel()[None, :]
        )
        indices = indices.reshape(shp).clip(0, None) # Note: a clip is performed because
                                                     # -1 is obtained in presence of NaN
        dist = dist.reshape(shp)

        return indices, dist

    def locate_index(values):
        raise NotImplementedError
    
    
class Locator_Regular(Locator):
    def __init__(
        self, coords, bounds: str, inversion_func: Callable | None = None, period=None
    ):
        self.inversion_func = inversion_func
        if inversion_func is None:
            self.coords = coords
        else:
            self.coords = inversion_func(coords)

        # Check that the coords are regular, otherwise raise a ValueError
        diff = np.diff(self.coords)
        self.is_time = 'timedelta' in diff.dtype.name
        if self.is_time:
            regular = np.allclose(diff[0], diff, rtol=1e-5, atol=np.timedelta64(0, "s"))
        else:
            regular = np.allclose(diff[0], diff)
        if not regular:
            raise ValueError

        self.vstart = self.coords[0]
        self.vend = self.coords[-1]
        if self.vstart < self.vend:
            self.vmin = self.vstart
            self.vmax = self.vend
        else:
            self.vmin = self.vend
            self.vmax = self.vstart
        self.N = len(self.coords)
        self.bounds = bounds

        deltat = (self.vend - self.vstart)
        if self.is_time:
            deltat = deltat / np.timedelta64(1, "s")
        self.scal = (self.N - 1) / deltat

        if (period is not None) and (not np.isclose(period, self.N / self.scal)):
            raise AssertionError(
                f"Expected a period of {period}, but a period of {self.N/self.scal} "
                "has been derived from the coords"
            )
    
    def locate_index_weight(self, values):
        """
        Optimized version of locate_index_weight for regular grid coords
        """
        if self.inversion_func is not None:
            values = self.inversion_func(values)

        values = self.handle_oob(values)
        
        # calculate floating index (scale to [0, N-1])
        if self.is_time:
            x = (values - self.vstart) * self.scal / np.timedelta64(1, "s")
        else:
            x = (values - self.vstart) * self.scal
        i_inf = np.floor(np.nan_to_num(x)).astype("int")

        dist = x - i_inf

        if self.bounds != "cycle":
            # Deals with indexing the last coordinate item
            mask_idk = i_inf == self.N - 1
            i_inf = np.where(mask_idk, i_inf - 1, i_inf)
            dist = np.where(mask_idk, np.float64(1), dist)
        
        return i_inf, dist
    
    #For Nearest ?
    def locate_index(values):
        raise NotImplementedError


def create_locator(
    coords, bounds: str, spacing, period: float | None = None
) -> Locator:
    """
    Locator factory    

    The purpose of this method is to instantiate the appropriate "Locator" class.

    The args are passed from the indexers.
    """
    if spacing != "irregular":
        # regular, auto, or callable
        ifunc = spacing if callable(spacing) else None
        try:
            return Locator_Regular(coords, bounds, inversion_func=ifunc, period=period)
        except ValueError:
            if spacing == "auto":
                pass
            else: # regular or callable: raise the exception
                raise

    return Locator(coords, bounds)


class Spline:
    def __init__(
        self, 
        values, 
        tension = 0.5, 
        bounds: Literal["error", "nan", "clip"] = "error", 
        spacing: Literal["regular", "irregular", "auto"]|Callable[[float],float] = "auto", 
    ):
        """
        A proxy class for Spline indeting.

        The purpose of this class is to provide a convenient interface to the
        interp function, by initializing an indexer class.

        Args:
            bounds (str): how to deal with out-of-bounds values:
                - error: raise a ValueError
                - nan: replace by NaNs
                - clip: clip values to the extrema
            spacing: how to deal with regular grids
                - "regular": assume a regular grid, raise an error if not
                - "irregular": assume an irregular grid
                - "auto": detect if grid is regular or not
                - if a function is provided, it is assumed that the grid is regular
                after applying this function. 
                (for example if the coords follow x² you need to feed sqrt)
            tension (float): how tight the rope between points is (1 very tight, 0 very loose)
                - 0: Very tight rope
                - 1: Very lose rope
                The points between indexes 0 and 1 and N - 1 and N will have a tightness of 0.5
        """
        self.values = values
        self.bounds = bounds
        self.tension = tension
        self.spacing = spacing

    def get_indexer(self, coords: xr.DataArray):
        cval = coords.values
        if len(cval) < 3 :
            raise ValueError("You need to have at least 3 points to spline in between") #AJOUTER LA BONNE ERREUR
        return Spline_Indexer(cval, self.bounds, self.spacing, self.tension)


class Spline_Indexer:
    def __init__(self, coords: NDArray, bounds: str, spacing, tension: float):
        self.coords = coords
        self.spacing = spacing
        self.bounds = bounds

        self.tension_matrix_full = np.array([
            [0,             1,              0,                  0],
            [-tension,      0,              tension,            0],
            [2 * tension,   -5 * tension,   4 * tension,        -tension],
            [-tension,      3 * tension,    -3 * tension,       tension],
        ])
        
        self.tension_matrix_early = np.array([
            [1, 0, 0, 0],
            [-1.5, 2, -0.5, 0],
            [0.5, -1, 0.5, 0],
            [0, 0, 0, 0],
        ])
        
        self.tension_matrix_late = np.array([
            [0, 1, 0, 0],
            [-0.5, 0, 0.5, 0],
            [0.5, -1, 0.5, 0],
            [0, 0, 0, 0],
        ])

        self.locator = create_locator(
            coords=self.coords,
            bounds=self.bounds,
            spacing=self.spacing,
        )
        
    def __call__(self, values):
        """
        Find indices of values for linear interpolation in self.coords

        Returns a list of tuples [(idx_inf, weights), (idx_sup, weights)]
        """
        
        N_min_1, dist = self.locator.locate_index_weight(values)
        
        N = len(self.coords)
        
        #init indices (-2 -1 +1 +2)
        indices = np.array([
            np.clip(N_min_1 - 1, -1, N), 
            N_min_1,
            np.clip(N_min_1 + 1, 0, N),
            np.clip(N_min_1 + 2, 0, N),
        ])
        
        mask_first_negative = indices[0] < 0 #early (-1 +1 +2)
        
        mask_last_oob = indices[3] > (N - 1) #late (-2 -1 +1)

        mask_normal = ~(mask_first_negative | mask_last_oob) #middle (-2 -1 +1 +2)
        
        # Initialize weights matrix
        weights = np.zeros((self.tension_matrix_full.shape[1], *dist.shape))
        # Calculate and assign weights based on masks
        if mask_normal.any():
            t_normal = dist[mask_normal]
            x_vector_full = np.vstack((np.ones_like(t_normal), t_normal, t_normal ** 2, t_normal ** 3)).T
            weights[:, mask_normal] = np.dot(self.tension_matrix_full.T, x_vector_full.T)
        
        if mask_first_negative.any():
            t_early = dist[mask_first_negative]
            x_vector_half_early = np.vstack((np.ones_like(t_early), t_early, t_early ** 2, np.zeros_like(t_early))).T
            weights[:, mask_first_negative] = np.dot(self.tension_matrix_early.T, x_vector_half_early.T)
        
        if mask_last_oob.any():
            t_late = dist[mask_last_oob]
            x_vector_half_late = np.vstack((np.ones_like(t_late), t_late, t_late ** 2, np.zeros_like(t_late))).T
            weights[:, mask_last_oob] = np.dot(self.tension_matrix_late.T, x_vector_half_late.T)

        #we change indices like we changed the weights
        if len(indices.shape) == 1:
            for i in range(1, 4):
                if(mask_first_negative):
                    indices[i - 1] = indices[i]
            if(mask_last_oob):
                 indices[3] = 0
        else:
            for i in range(1, 4):
                indices[i - 1][mask_first_negative] = indices[i][mask_first_negative]
            indices[3][mask_last_oob] = 0
        
            
        return [(indices[i], weights[i]) for i in range(len(weights))]
    
        

class Linear:
    def __init__(
        self,
        values: xr.DataArray,
        bounds: Literal["error", "nan", "clip", "cycle"] = "error",
        spacing: Literal["regular", "irregular", "auto"]
        | Callable[[float], float] = "auto",
        period: float | None = None,
    ):
        """
        A proxy class for Linear indexing.

        The purpose of this class is to provide a convenient interface to the
        interp function, by initializing an indexer class.

        Args:
            bounds (str): how to deal with out-of-bounds values:
                - "error": raise a ValueError
                - "nan": replace by NaNs
                - "clip": clip values to the extrema
                - "cycle": the axis is considered cyclic
                    e.g.: longitudes between -180 and 179
                    allows indexing values in the range of [-180, 180] or even [0, 360].
                    The "period" argument can be used to check the period inferred
                    from the coord values, for example period=360.
            spacing: how to deal with regular grids
                - "regular": assume a regular grid, raise an error if not
                - "irregular": assume an irregular grid
                - "auto": detect if grid is regular or not
                - if a function is provided, it is assumed that the grid is regular
                  after applying this function. 
                  (for example if the coords follow x² you need to feed sqrt)
                The index lookup is optimized for regular grids, using the
                Locator_Regular class.
            period (optional, float): if provided, we verify that the period inferred
                from the coordinates is equal to this value.
                (only when bounds="cycle")
        """
        self.values = values
        self.bounds = bounds
        self.spacing = spacing
        self.period = period
    
    def get_indexer(self, coords: xr.DataArray):
        cval = coords.values
        if len(cval) < 2 :
            raise ValueError(
                f"Cannot apply linear indexing to an axis of {len(cval)} values"
            )
        return Linear_Indexer(cval, self.bounds, self.spacing, self.period)


class Linear_Indexer:
    def __init__(
        self, coords: NDArray, bounds: str, spacing, period=None
    ):
        self.bounds = bounds
        self.N = len(coords)
        self.coords = coords
        self.spacing = spacing
        self.period = period

        # check ascending/descending order
        diff = np.diff(coords).astype(float)
        zero = np.zeros(1, dtype=diff.dtype)
        if (diff > zero).all():
            self.ascending = True
            self.coords = coords
        elif (diff < zero).all():
            self.ascending = False
            self.coords = coords[::-1].copy()
        else:
            raise ValueError('Input coords should be monotonous.')

        self.locator = create_locator(
            coords=self.coords,
            bounds=self.bounds,
            spacing=self.spacing,
            period=period,
        )
        
        # print(self.locator)
        

    def __call__(self, values: NDArray) -> List:
        """
        Find indices of `values` for linear interpolation in self.coords

        Returns a list of tuples [(idx_inf, weights), (idx_sup, weights)]
        """
        
        indices, dist = self.locator.locate_index_weight(values)

        if self.ascending:
            iinf = indices
            winf = 1 - dist
            isup = indices + 1
            wsup = dist
        else:
            iinf = self.N - 1 - indices
            isup = self.N - 2 - indices
            winf = 1 - dist
            wsup = dist
        
        if self.bounds == "cycle":
            iinf = iinf % self.N
            isup = isup % self.N

        return [(iinf, winf), (isup, wsup)]


class Index:
    def __init__(self, values):
        """
        Proxy class for integer index-based selection (isel)

        Does not require a separate Indexer class, since the coords are not used;
        the values are returned as-is, they are already integer indices.
        """
        self.values = values
    
    def get_indexer(self, coords):
        # We can simply return self to avoid another class, since coords are not used.
        # The __call__ method of the Indexer is implemented in this class.
        return self
        
    def __call__(self, values: NDArray):
        return [(values, None)]


class Nearest:
    def __init__(
        self,
        values: xr.DataArray,
        tolerance: float | None = None,
        spacing: Literal["auto"] | Callable[[float], float] = "auto",
    ):
        """
        Proxy class for value selection (sel)

        Args:
            values (xr.DataArray): values for selection
            tolerance (float, optional): absolute tolerance for inexact search
            spacing(str | Callable): 
                - "auto" : will take the value of the nearest valid value
                - lambda x: f(x) : will take the value of the nearest valid value after
                    inverting the x axis values based on lambda
        """
        self.values = values
        self.tolerance = tolerance
        self.spacing = spacing
    
    def get_indexer(self, coords: xr.DataArray):
        return Nearest_Indexer(coords.values, self.tolerance, self.spacing)


class Nearest_Indexer:
    def __init__(self, coords: NDArray, tolerance: float|None, spacing: str|Callable = "auto"):
        self.tolerance = tolerance
        
        if isinstance(spacing, str) or isinstance(spacing, Callable) :
            TypeError("Spacing is of the wrong type, waiting for str or Callable")
            
        self.spacing = spacing
        if (np.diff(coords) > 0).all():
            self.ascending = True
            self.coords = coords
        elif (np.diff(coords) < 0).all():
            self.ascending = False
            self.coords = coords[::-1]
        else:
            raise ValueError('Input coords should be monotonous.')
    
    def __call__(self, values: NDArray):
        mvalues = None
        if self.spacing == "auto":
            mvalues = values
            coords = self.coords
        elif isinstance(self.spacing, Callable):
            mvalues = self.spacing(values)
            coords = self.spacing(self.coords)
        else:
            ValueError("spacing isn't 'auto' or a Callable (lambda x: f(x))")
        
        
        idx = np.searchsorted(coords, mvalues)
        idx = idx.clip(0, len(coords) - 1)

        # distance to the inf/sup bounds
        dist_inf = np.abs(mvalues - coords[idx-1]) # NOTE: EDGE CASE TO BE HANDLED ? [idx = 0]
        dist_sup = np.abs(coords[idx] - mvalues)
        
        # NOTE: EDGE CASE HANDLING ?
        mask_idx_neg = (idx -1) < 0

        dist_inf = np.where(mask_idx_neg, dist_sup + 1, dist_inf)


        if (self.tolerance is not None) and (
            np.minimum(dist_inf, dist_sup) > self.tolerance
        ).any():
            raise ValueError

        idx_closest = np.where(dist_inf < dist_sup, idx-1, idx)

        if self.ascending:
            return [(idx_closest, None)]
        else:
            return [(len(coords) - 1 - idx_closest, None)]


def broadcast_numpy(ds: xr.Dataset) -> Dict:
    """
    Returns all data variables in `ds` as numpy arrays
    broadcastable against each other
    (with new single-element dimensions)

    This requires the input to be broadcasted to common dimensions.
    """
    result = {}
    for var in ds:
        result[var] = ds[var].data[
            tuple([slice(None) if d in ds[var].dims else None for d in ds.dims])
        ]
    return result


def broadcast_shapes(ds: xr.Dataset, dims) -> Dict:
    """
    For each data variable in `ds`, returns the shape for broadcasting
    in the dimensions defined by dims
    """
    result = {}
    for var in ds:
        result[var] = tuple(
            [
                ds[var].shape[ds[var].dims.index(d)] if d in ds[var].dims else 1
                for d in dims
            ]
        )
    return result


def determine_output_dimensions(
    data: xr.DataArray, ds_dims: list, dims_sel_interp: Iterable
) -> list:
    """
    Determine output dimensions for interpolated/selected DataArray.
    
    This function implements numpy's advanced indexing rules to determine the final
    dimension ordering of the output DataArray after interpolation/selection operations.
    
    The key principle is that when advanced indexing is applied to some dimensions
    of an array, those dimensions are replaced by the dimensions of the indexing arrays,
    and these new dimensions are inserted at the position of the first indexed dimension.
    
    Args:
        data (xr.DataArray): The input DataArray being interpolated/selected
        ds_dims (list): List of dimension names from the indexing Dataset (the new
            dimensions that will replace the indexed dimensions)
        dims_sel_interp (set or list): Set/list of dimension names from `data` that
            are being interpolated or selected (i.e., dimensions with advanced indexing)
    
    Returns:
        list: Ordered list of dimension names for the output DataArray
    """
    out_dims = []
    dims_added = False
    for dim in data.dims:
        if dim in dims_sel_interp:
            if not dims_added:
                out_dims.extend(list(ds_dims))
                dims_added = True
        else:
            out_dims.append(dim)
    
    return out_dims
