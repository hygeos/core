from datetime import datetime
from functools import reduce
from itertools import product
from typing import Dict, Iterable, List, Literal, Callable

import numpy as np
import xarray as xr
from numpy.typing import NDArray
from packaging import version
# from scipy.interpolate._rgi_cython import find_indices

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



def interp(da: xr.DataArray, *, method=1, **kwargs):
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

    # transpose ds to get fixed dimension ordering
    ds = ds.transpose(*ds.dims)
    
    out_dims = determine_output_dimensions(da, ds, kwargs.keys())

    ret = xr.map_blocks(
        interp_block,
        ds,
        kwargs={
            "da": da,
            "out_dims": out_dims,
            "indexers": indexers,
            "method": method,
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
    ds: xr.Dataset, da: xr.DataArray, out_dims, indexers: Dict, method: int
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
    data_values = da.values
    if method == 1:
        result = 0
        for iw in product_dict(**indices_weights):
            weights = [
                1 if iw[dim][1] is None else iw[dim][1].reshape(w_shape[dim])
                for dim in iw
            ]
            w = reduce(lambda x, y: x*y, weights)
            keys = [(iw[dim][0] if dim in iw else slice(None)) for dim in da.dims]
            result += data_values[tuple(keys)] * w
    else:
        # reformat indices_weights as a list for each dimension in da
        # TODO: use a single indices_weights for the above as well
        indices_weights_list = [
            indices_weights[dim]
            if dim in indices_weights
            else (np.array([]), np.array(1.0))   # non-interpolated dimensions
            for dim in da.dims
        ]
        if method == 2:
            # convert to numba list, otherwise it does not get passed to numba
            # warning, timing includes numba compilation
            result = interpolate_numba_method2(data_values, List(indices_weights_list))
        elif method == 3:
            result = interpolate_numba_method3(data_values, List(indices_weights_list))
        else:
            raise ValueError(f"Invalid method {method}")
    
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
def find_indices(coordinates, values) -> tuple[np.ndarray, np.ndarray]:
    """
    assumes the coordinates is sorted and one dim
    assumes the values array is flattened
        -> require reshaping after call
        
    Note:
        When extrapolating the distance is computing by assuming the same distance between the previous
        valid segment to extrapolate the missing point.
        
        This function reproduces the behavior of scipy interpolate._rgi_cython.find_indices
        
        ex:
            coords = [1,2,4,8]
            find_indices(coords, [10])
            
            1, 2, 4, 8, (10), ?? missing point
            
            8 -> 10 -> 12 (same max dist as previous segment 4 - 8)
            dist = 0.5
            
            This behavior is a best guess, as obivously in our case logic would be that the bound should be 16, 
            but this cannot be infered rigourously.
         
    """
    
    coordinates = coordinates[0].flatten()
    values = values.flatten()
    
    left_indices   = np.empty(len(values), dtype=np.int32)    # not uint32: when extrapolating the index can be negative, and it overflows
    left_distances = np.empty(len(values), dtype=np.float32)  # not float64: distances are roughly comprised between [0 - 1] (excepted when extrapolating)
    
    size = len(values) # store locally because used twise
    
    for i in prange(size): # parallel for loop OMP
        value = values[i]
        index = np.searchsorted(coordinates, value, side="left") - 1
        
        # NOTE: ---------------------------------
        #   * All the computations below could be done on the whole vector, outside the loop 
        #   * It could degrade perforamce because of cache locality.
        #   * It could allow for vectorized assembly instructions (way faster) which may not be possible in this loop
        #   * It would be faster when numba is not installed (no jit -> empty decorator)
        #   because of the searchsorted.
        #   * search sorted is probably the bottleneck anyway
        # ---------------------------------------
        
        # : inline 2 next line onto previous ? (compiler could already be optimizing it anyway..)
        offset = (index < 0) + (index >= size) * -1     # if extrapolating leftside: 1, if extrapolating rightside: -1, otherwise: 0
        index += offset                                 # if extrapolating, bring the index back to the bounds
        
        left_indices[i] = index                         # store left index
        left_val  = coordinates[index]                  # store locally leftbound val because used twice
        right_val = coordinates[index + 1]              # store locally rightbound val because used twice
        left_distances[i] = (value - left_val) / (right_val - left_val) # compute distance
        
    return (left_indices[None, :], left_distances[None, :])

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
        tolerance: float | None = 1e-8,
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
        dist_inf = np.abs(mvalues - coords[idx-1]) # EDGE CASE TO BE HANDLED ? [idx = 0]
        dist_sup = np.abs(coords[idx] - mvalues)
        
        #EDGE CASE HANDLING ?
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


def determine_output_dimensions(data, ds, dims_sel_interp):
    """
    determine output dimensions
    based on numpy's advanced indexing rules
    """
    out_dims = []
    dims_added = False
    for dim in data.dims:
        if dim in dims_sel_interp:
            if not dims_added:
                out_dims.extend(list(ds.dims))
                dims_added = True
        else:
            out_dims.append(dim)
    
    return out_dims


@njit
def inc_coords(pos, shp):
    """
    Increment integer coordinates defined by vector `pos` within dimensions of
    given shape `shp`
    """
    pos[0] += 1
    for i in range(len(shp)-1):
        if pos[i] == shp[i]:
            pos[i] = 0
            pos[i+1] += 1
        else:
            break


@njit
def muls(a, b):
    """ 
    Multiply all elements of `a` by a scalar `b` (in place)
    """
    af = a.ravel()
    for i in range(af.size):
        af[i] *= b

@njit
def mul(a: NDArray, b: NDArray, inplace: bool) -> NDArray:
    """
    Multiply `a` and `b` element by element
    """
    assert a.shape == b.shape
    if not inplace:
        out = a.copy()
    else:
        out = a

    outf = out.ravel()
    bf = b.ravel()
    for i in range(outf.size):
        outf[i] *= bf[i]
    
    return out

@njit
def add(a: NDArray, b: NDArray, inplace: bool) -> NDArray:
    """
    Add `a` and `b` element by element
    """
    assert a.shape == b.shape
    if inplace:
        out = a
    else:
        out = a.copy()
    outf = out.ravel()
    bf = b.ravel()
    for i in range(outf.size):
        outf[i] += bf[i]


@njit
def interpolate_numba_method2(data: NDArray, indices_weights: list) -> NDArray:
    """
    data is the array to interpolate

    indices_weights is a list (for each `data` dimension) of list of tuples
        (indices, weights)
        where `indices` and `weights` are the indices and weights to apply
        along each diumension.
        if indices is an array of size 0 (eg, np.array([])), then the dimension is
        left as-is (eg, slice(None)).
    """
    data_flat = data.ravel()
    assert data.flags.c_contiguous

    # get the number of points to iterate along each dimension
    # (shape of the hypercube)
    shph = [len(ls) for ls in indices_weights]
    shp = data.shape
    ndim = len(shp)
    assert len(shph) == ndim
    
    # get the total number of items to iterate (cartesian product)
    sizeh = 1
    for t in shph:
        sizeh *= t
    
    pos = [0 for _ in shph] # current position in the hypercube

    out = None
    
    # loop over the points in the hypercube
    for _ in range(sizeh):
        # calculate current address in the flat data array
        # and the associated weight
        # (loop over the dimensions)
        addr = None
        weight = None
        
        for i in range(ndim):
            # tuple of indice+weight for current dimension
            iw = indices_weights[i][pos[i]]
            ind = iw[0]
            w = iw[1]
            
            if addr is None:
                addr = ind.copy()
                addr_flat = addr.ravel()
            else:
                #addr = addr*shp[i+1] + ind
                muls(addr, shp[i])
                add(addr, ind, True)
            
            if weight is None:
                weight = w.copy()
                wf = weight.ravel()
            else:
                mul(weight, w, True)

        # accumulate (data_flat[addr] * weight) in `out`
        if out is None:
            out = np.zeros(weight.shape, dtype='float64')
            outf = out.ravel()
        for i in range(outf.size):
            outf[i] += data_flat[addr_flat[i]] * wf[i]

        # increment the position in the hypercube
        inc_coords(pos, shph)
        
    return out


@njit
def interpolate_numba_method3(data: NDArray, indices_weights: list):
    """
    Other tentative version of interpolate_numba where the interpolation operates on a
    scalar (interpolate_numba_method3_inner)

    This function loops over all elements of the final array
    
    Essentially loops over the output grid and compute the hypercube weighted mean from the data grid
    """
    # for coord in <pixel loop>:
    #     interpolate_numba_method3_inner(data, indices_weights_scal)
    raise NotImplementedError


@njit
def interpolate_numba_method3_inner(
    data: NDArray, indices_weights_scal: list,
) -> float:
    """
    Interpolation of a scalar in `data`

    indices_weights_scal: list (for each dimension) of lists (for each combination of
    points) of tuples (index, weight)
    """
    raise NotImplementedError
