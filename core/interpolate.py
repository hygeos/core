import warnings
from functools import reduce
from itertools import product
from typing import Any, Dict, Iterable, List, Literal, Optional, Callable

import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import NDArray
from packaging import version
from scipy.interpolate._rgi_cython import find_indices


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
        - Supports per-dimension options (nearest neighbour selection, interpolation,
          out-of-bounds behaviour...)

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
        ...          bounds='clip'),  # clip out of bounds values to the axis min/max.
        ...     b = Nearest(b_values), # perform nearest neighbour selection along
        ...                            # dimension `a`; `b_values` is a DataArray
        ...                            # with dimension (x, y)
        ... ) # returns a DataArray with dimensions (x, y, c)
        No interpolation or selection is performed along dimension `c` thus it is
        left as-is.

    Returns:
        xr.DataArray: DataArray on the new coordinates.
    """
    if set(kwargs).issubset(set(['sel', 'interp', 'options'])):
        # interp version 1
        warnings.warn('This is a backward comparible version of the interp function. '
                      'Please use the updated API (see interp docstring).')
        return interp_v1(da, **kwargs)
    else:
        # interp version 2
        return interp_v2(da, **kwargs)


def interp_v2(da: xr.DataArray, **kwargs) -> xr.DataArray:
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
        interp_block_v2,
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


def interp_block_v2(
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
    indices_weights = {k: v(np_indexers[k]) for k, v in indexers.items()}

    # cartesian product of the combination of lower and upper indices (in case of
    # linear interpolation) for each dimension
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

    return ret


class Locator:
    def __init__(self, coords: NDArray, bounds: str):
        self.coords = coords.astype("double")
        self.bounds = bounds
        pass
    
    #For Linear and Spline
    def locate_index_weight(self, values):
        """
        Find indices and dist of values for linear and spline interpolation in self.coords

        Returns a list of indices, dist (float 0 to 1) and oob
        """
        shp = values.shape
        indices, dist = find_indices(
            (self.coords,), values.astype("double").ravel()[None, :]
        )
        indices = indices.reshape(shp)
        dist = dist.reshape(shp)
        oob = None
        if self.bounds == "clip":
            dist = dist.clip(0, 1)
        else:
            oob = (dist < 0) | (dist > 1)
            if self.bounds == "error" and oob.any():
                raise ValueError
            elif self.bounds == "nan":
                dist[oob] = np.NaN
        
        return indices, dist, oob

    
    #For Nearest
    def locate_index(values):
        raise NotImplementedError
        
    
    
class Locator_Regular(Locator):
    def __init__(self, vmin: float, vmax: float, N: int, bounds: str):
        self.vmin = vmin
        self.vmax = vmax
        self.N = N
        self.bounds = bounds
        self.scal = (self.N - 1) / (self.vmax - self.vmin)
        pass
    
    #For Linear and Spline
    def locate_index_weight(self, values):
        # floating index (scale to [0, N-1])
        x = (values - self.vmin) * self.scal
        oob = None
        # out of bounds management
        if self.bounds == "clip":
            x = x.clip(0, self.N - 1)
        else:
            oob = (x < 0) | (x > self.N - 1)
            
            
        i_inf = np.floor(x).astype("int").clip(0, self.N - 1)
        dist = x - i_inf
        
        if np.isscalar(x): #made to handle scalars
            # Apply mask condition if `i_inf == N - 1` for scalar
            if i_inf == self.N - 1:
                i_inf -= 1
                dist = 1
        else:
            # Apply mask condition for array
            mask_idk = i_inf == self.N - 1
            i_inf[mask_idk] -= 1
            dist[mask_idk] = 1
        
        return i_inf, dist, oob
    
    #For Nearest ?
    def locate_index(values):
        raise NotImplementedError


class Locator_Inversed_Func:
    def __init__(self, coords: NDArray, bounds: str, inversion_func):
        self.coords = coords.astype("double")
        self.bounds = bounds
        self.inversion_func = inversion_func
        pass
    
    #For Linear and Spline
    def locate_index_weight(self, values):
        """
        Find indices and dist of values for linear and spline interpolation in self.coords

        Returns a list of indices, dist (float 0 to 1) and oob
        """
        x = self.inversion_func(values)
        N = len(self.coords)
        
        oob = None
        # out of bounds management
        if self.bounds == "clip":
            x = x.clip(0, N - 1)
        else:
            oob = (x < 0) | (x > N - 1)
            
            
        i_inf = np.floor(x).astype("int").clip(0, N - 1)
        dist = x - i_inf
        
        if np.isscalar(x): #made to handle scalars
            # Apply mask condition if `i_inf == N - 1` for scalar
            if i_inf == N - 1:
                i_inf -= 1
                dist = 1
        else:
            # Apply mask condition for array
            mask_idk = i_inf == N - 1
            i_inf[mask_idk] -= 1
            dist[mask_idk] = 1
        
        return i_inf, dist, oob


    #For Nearest ?
    def locate_index(values):
        raise NotImplementedError
        

def create_locator(coords, bounds: str, regular: str, inversion_func) -> Locator:
    """
        The purpose of this methode is to generate the correct Locator based on the coords, the bounds, if it's regular and the inversion func
        so far it can produce a Locator, a Locator_Regular and a Locator_Inversed_Func

        Args:
            bounds (str): how to deal with out-of-bounds values:
                - error: raise a ValueError
                - nan: replace by NaNs
                - clip: clip values to the extrema
            regular (str): how to deal with regular grids
                - yes: raise an error in case of non-regular grid
                - no: disable regular grid detection
                - auto: detect if grid is regular or not
            inversion_fun: is the inverse of the function the indexes uses (for example if indexes follow x² you need to feed sqrt(x)) in a lambda form : lambda x: np.sqrt(x) 
    """
    # factory pour Locator ou Locator_Regular
    
    if inversion_func != None :
        return Locator_Inversed_Func(coords, bounds, inversion_func)
    
    # regular grid detection
    cval = coords
    
    if regular in ['regular', 'auto']:
        diff = np.diff(cval)
        regular = np.allclose(diff[0], diff)
        if regular == 'regular':
            assert regular
    else:
        regular = False
        
    if regular:
        return Locator_Regular(cval[0], cval[-1], len(cval), bounds)
    else:
        return Locator(cval, bounds)


class Spline:
    def __init__(
        self, 
        values, 
        tension = 0.5, 
        bounds: Literal["error", "nan", "clip"] = "error", 
        spacing: Literal["regular", "irregular", "auto"]|Callable[[float],float] = "auto", 
        ):
        self.values = values
        self.bounds = bounds
        self.tension = tension
        
        self.regular = None
        self.inversion_func = None
        
        if isinstance(spacing, str) :
            self.regular = spacing
        elif isinstance(spacing, Callable):
            self.inversion_func = spacing
        
        if self.regular == None and self.inversion_func == None:
            TypeError("spacing should either be a str or a Callable")
            
    """
        A proxy class for Spline indeting.

        The purpose of this class is to provide a convenient interface to the
        interp function, by initializing an indexer class.

        Args:
            bounds (str): how to deal with out-of-bounds values:
                - error: raise a ValueError
                - nan: replace by NaNs
                - clip: clip values to the extrema
            regular 
            (str): how to deal with regular grids
                - yes: raise an error in case of non-regular grid
                - no: disable regular grid detection
                - auto: detect if grid is regular or not
            (Callable) : is the inverse of the function the indexes uses (for example if indexes follow x² you need to feed sqrt(x)) in a lambda form : lambda x: np.sqrt(x) 
            tension (float): how tight the rope between points is (1 very tight, 0 very loose)
                - 0: Very tight rope
                - 1: Very lose rope
                The points between indexes 0 and 1 and N - 1 and N will have a tightness of 0.5
    """
    def get_indexer(self, coords: xr.DataArray):
        
        cval = coords.values
        if len(cval) < 3 : #2 1 0
            raise ValueError("You need to have at least 3 points to spline in between") #AJOUTER LA BONNE ERREUR
        return Spline_Indexer(cval, self.bounds, self.regular, self.tension, self.inversion_func)


class Spline_Indexer:
    def __init__(self, coords: NDArray, bounds: str, regular: str, tension: float, inversion_func = None):
        self.coords = coords
        self.regular = regular
        self.bounds = bounds
        self.inversion_func = inversion_func

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
        
        """
        A proxy class for Spline indeting.

        The purpose of this class is to provide a convenient interface to the
        interp function, by initializing an indexer class.

        Args:
            bounds (str): how to deal with out-of-bounds values:
                - error: raise a ValueError
                - nan: replace by NaNs
                - clip: clip values to the extrema
            regular 
            (str): how to deal with regular grids
                - yes: raise an error in case of non-regular grid
                - no: disable regular grid detection
                - auto: detect if grid is regular or not
            (Callable) : is the inverse of the function the indexes uses (for example if indexes follow x² you need to feed sqrt(x)) in a lambda form : lambda x: np.sqrt(x) 
            tension (float): how tight the rope between points is (1 very tight, 0 very loose)
                - 0: Very tight rope
                - 1: Very lose rope
                The points between indexes 0 and 1 and N - 1 and N will have a tightness of 0.5
        """
        
    def __call__(self, values):
        """
        Find indices of values for linear interpolation in self.coords

        Returns a list of tuples [(idx_inf, weights), (idx_sup, weights)]
        """
        
        locator = create_locator(coords=self.coords, bounds=self.bounds, regular=self.regular, inversion_func=self.inversion_func)
        
        N_min_1, dist, oob = locator.locate_index_weight(values)
        
        if self.bounds == "clip":
            dist = dist.clip(0, 1)
        else:
            oob = (dist < 0) | (dist > 1)
            if self.bounds == "error" and oob.any():
                raise ValueError
            elif self.bounds == "nan":
                dist[oob] = np.NaN
        
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
        for i in range(1, 4):
            indices[i - 1][mask_first_negative] = indices[i][mask_first_negative]
        indices[3][mask_last_oob] = 0
        
        
        
        if self.bounds != 'clip':
            if self.bounds == 'error':
                if oob.any():
                    raise ValueError
            elif self.bounds == "nan":
                for i in range(len(weights)) :
                    weights[i][oob] = np.NaN
            
        return [(indices[i], weights[i]) for i in range(len(weights))]
    
        

class Linear:
    def __init__(
        self,
        values: xr.DataArray,
        bounds: Literal["error", "nan", "clip"] = "error",
        spacing: Literal["regular", "irregular", "auto"]|Callable[[float],float] = "auto", 
    ):
        """
        A proxy class for Linear indeting.

        The purpose of this class is to provide a convenient interface to the
        interp function, by initializing an indexer class.

        Args:
            bounds (str): how to deal with out-of-bounds values:
                - error: raise a ValueError
                - nan: replace by NaNs
                - clip: clip values to the extrema
            regular (str): how to deal with regular grids
                - yes: raise an error in case of non-regular grid
                - no: disable regular grid detection
                - auto: detect if grid is regular or not
            inversion_fun: is the inverse of the function the indexes uses (for example if indexes follow x² you need to feed sqrt(x)) in a lambda form : lambda x: np.sqrt(x) 
        """
        self.values = values
        self.bounds = bounds
        
        self.regular = None
        self.inversion_func = None
        
        #I want to raise an error as fast as possible :
        
        if isinstance(spacing, str) :
            self.regular = spacing
        elif isinstance(spacing, Callable):
            self.inversion_func = spacing
        
        if self.regular == None and self.inversion_func == None:
            TypeError("spacing should either be a str or a Callable")
    
    def get_indexer(self, coords: xr.DataArray):
        cval = coords.values
        if len(cval) < 2 :
            raise ValueError("Linear needs at least 2 point to index properly") #AJOUTER LA BONNE ERREUR
        return Linear_Indexer(cval, self.bounds, self.regular, self.inversion_func)


class Linear_Indexer:
    def __init__(self, coords: NDArray, bounds: str, regular: str, inversion_func):
        self.bounds = bounds
        self.N = len(coords)
        self.coords = coords
        self.regular = regular
        self.inversion_func = inversion_func

        #check ascending/descending order
        if (np.diff(coords) > 0).all():
            self.ascending = True
            self.coords = coords
        elif (np.diff(coords) < 0).all():
            self.ascending = False
            self.coords = coords[::-1].copy()
        else:
            raise ValueError('Input coords should be monotonous.')

    def __call__(self, values: NDArray) -> List:
        """
        Find indices of `values` for linear interpolation in self.coords

        Returns a list of tuples [(idx_inf, weights), (idx_sup, weights)]
        """
        locator = create_locator(coords=self.coords, bounds=self.bounds, regular=self.regular, inversion_func=self.inversion_func)
        
        indices, dist, oob = locator.locate_index_weight(values)

        
        if self.bounds != 'clip':
            if self.bounds == 'error':
                if oob.any():
                    raise ValueError
            elif self.bounds == "nan":
                for i in range(len(dist)) :
                    dist[oob] = np.NaN
                    
        if self.ascending:
            return [(indices, 1 - dist), (indices + 1, dist)]
        else:
            return [
                (self.N - 1 - indices, 1 - dist),
                (self.N - 2 - indices, dist),
            ]


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
        self, values: xr.DataArray, tolerance: float|None = 1e-8,
    ):
        """
        Proxy class for value selection (sel)

        Args:
            values (xr.DataArray): values for selection
            tolerance (float, optional): absolute tolerance for inexact search
        """
        self.values = values
        self.tolerance = tolerance
    
    def get_indexer(self, coords: xr.DataArray):
        return Nearest_Indexer(coords.values, self.tolerance)

class Nearest_Indexer:
    def __init__(self, coords: NDArray, tolerance: float|None):
        self.tolerance = tolerance
        if (np.diff(coords) > 0).all():
            self.ascending = True
            self.coords = coords
        elif (np.diff(coords) < 0).all():
            self.ascending = False
            self.coords = coords[::-1]
        else:
            raise ValueError('Input coords should be monotonous.')
    
    def __call__(self, values: NDArray):
        idx = np.searchsorted(self.coords, values).clip(0, len(self.coords) - 1)

        # distance to the inf/sup bounds
        dist_inf = np.abs(values - self.coords[idx-1])
        dist_sup = np.abs(self.coords[idx] - values)

        if (self.tolerance is not None) and (
            np.minimum(dist_inf, dist_sup) > self.tolerance
        ).any():
            raise ValueError

        idx_closest = np.where(dist_inf < dist_sup, idx-1, idx)

        if self.ascending:
            return [(idx_closest, None)]
        else:
            return [(len(self.coords) - 1 - idx_closest, None)]



def interp_v1(
    da: xr.DataArray,
    *,
    sel: Optional[Dict[str, xr.DataArray]] = None,
    interp: Optional[Dict[str, xr.DataArray]] = None,
    options: Optional[Dict[str, Any]] = None,
) -> xr.DataArray:
    """Interpolate or select a DataArray onto new coordinates

    This function is similar to xr.interp and xr.sel, but:
        - Supports dask-based inputs (in sel and interp) without
          triggering immediate computation
        - Supports both selection and indexing
        - Does not use xarray's default .interp method (improved efficiency)

    Parameters
    ----------
    da : xr.DataArray
        The input DataArray
    sel : Optional[Dict[str, xr.DataArray]], optional
        A dict mapping dimensions names to values, by default None
    interp : Optional[Dict[str, xr.DataArray]], optional
        A dict mapping dimension names to new coordinates, by default None
    options : Optional[Dict[str, Any]], optional
        A dict mapping dimension names to a dictionary of options to use for
        the current sel or interp.
        For sel dimensions, the options are passed to `pandas.Index.get_indexer`. In
        particular, 'method' can be one of:
            - None [default, raises an error if values not in index]
            - "pad"/"ffill"
            - "backfill"/"bfill"
            - "nearest"
        For interp dimensions:
            `bounds`: behaviour in case of out-of-bounds values (default "error")
                - "error": raise an error
                - "nan": set NaN values
                - "clip": clip values within the bounds
            `skipna`: whether to skip input NaN values (default True)

    Example
    -------
    >>> interp(
    ...     data,  # input DataArray with dimensions (a, b, c)
    ...     interp={ # interpolation dimensions
    ...         'a': a_values, # `a_values` is a DataArray with dimension (x, y)
    ...     },
    ...     sel={ # selection dimensions
    ...         'b': b_values, # `b_values` is a DataArray with dimensions (x)
    ...     },
    ...     options={ # define options per-dimension
    ...         'a': {"bounds": "clip"},
    ...         'b': {"method": "nearest"},
    ...     },
    ... ) # returns a DataArray with dimensions (x, y, c)
    No interpolation or selection is performed along dimension `c` thus it is
    left as-is.

    Returns
    -------
    xr.DataArray
        New DataArray on the new coordinates.
    """
    assert version.parse(xr.__version__) >= version.parse("2024.01.0")

    assert (da.chunks is None) or (
        len(da.chunks) == 0
    ), "Input DataArray should not be dask-based"

    sel = sel or {}
    interp = interp or {}

    # group all sel+interp dimensions
    ds = xr.Dataset({**sel, **interp})

    # prevent common dimensions between da and sel+interp
    assert not set(ds.dims).intersection(da.dims)

    # transpose them to ds.dims
    ds = ds.transpose(*ds.dims)

    dims_sel_interp = list(sel.keys()) + list(interp.keys())
    out_dims = determine_output_dimensions(da, ds, dims_sel_interp)

    ret = xr.map_blocks(
        index_block,
        ds,
        kwargs={
            "data": da,
            "dims_sel": sel.keys(),
            "dims_interp": interp.keys(),
            "options": options,
            "out_dims": out_dims,
        },
    )
    ret.attrs.update(da.attrs)

    return ret


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


def index_block(
    ds: xr.Dataset,
    data: xr.DataArray,
    dims_sel: List,
    dims_interp: List,
    out_dims: List,
    options: Optional[Dict] = None,
) -> xr.DataArray:
    """
    This function is called by map_blocks in function `interp`, and performs the
    indexing and interpolation at the numpy level.

    NOTE: to be deprecated in favour of interp_v2 and interp_block_v2
    """
    options = options or {}

    # get broadcasted data from ds (all with the same number of dimensions)
    np_indexers = broadcast_numpy(ds)
    x_indexers = broadcast_shapes(ds, out_dims)

    keys = [slice(None)] * data.ndim

    # selection keys (non-interpolation dimensions)
    for dim in dims_sel:
        idim = data.dims.index(dim)

        # default sel options
        opt = {**(options[dim] if dim in options else {})}
        keys[idim] = (
            data.indexes[dim]
            .get_indexer(np_indexers[dim].ravel(), **opt)
            .reshape(np_indexers[dim].shape)
        )
        if ((keys[idim]) < 0).any():  # type: ignore
            raise ValueError(
                f"Error in selection of dimension {dim} with options={opt} in "
                f'interpolation of DataArray "{data.name}"'
            )

    # determine bracketing values and interpolation ratio
    # for each interpolation dimension
    iinf = {}
    x_interp = {}
    for dim in dims_interp:
        # default interp options
        opt = {
            "bounds": "error",
            "skipna": True,
            **(options[dim] if dim in options else {}),
        }
        assert opt["bounds"] in ["error", "nan", "clip"]

        iinf[dim] = (
            data.indexes[dim]
            .get_indexer(np_indexers[dim].ravel(), method="ffill")
            .reshape(np_indexers[dim].shape)
        )

        # Clip indices
        iinf[dim] = iinf[dim].clip(0, len(data.indexes[dim]) - 2)

        iinf_dims_out = iinf[dim].reshape(x_indexers[dim])
        vinf = data.indexes[dim].values[iinf_dims_out]
        vsup = data.indexes[dim].values[iinf_dims_out + 1]
        x_interp[dim] = np.array(
            np.clip(
                (np_indexers[dim].reshape(x_indexers[dim]) - vinf) / (vsup - vinf), 0, 1
            )
        )

        # skip nan values
        if opt["skipna"]:
            isnan = np.isnan(np_indexers[dim])
        else:
            isnan = np.array(False)

        # deal with out-of-bounds (non-nan) values
        if opt["bounds"] in ["error", "nan"]:
            valid = in_index(data.indexes[dim], np_indexers[dim])

            if opt["bounds"] == "error":
                if (~valid & ~isnan).any():
                    vmin = np_indexers[dim][~isnan].min()
                    vmax = np_indexers[dim][~isnan].max()
                    raise ValueError(
                        f"out of bounds values [{vmin} -> {vmax}] during interpolation "
                        f"of {data.name} in dimension {dim} "
                        f"[{data.indexes[dim][0]}, {data.indexes[dim][-1]}] "
                        f"with options={opt}"
                    )

            # opt['bounds'] == "nan"
            x_interp[dim][(~valid | isnan).reshape(x_indexers[dim])] = np.NaN

        elif opt["skipna"]:
            x_interp[dim][isnan.reshape(x_indexers[dim])] = np.NaN

    # loop over the 2^n bracketing elements
    # (cartesian product of [0, 1] over n dimensions)
    result = 0
    data_values = data.values
    for b in range(2 ** len(dims_interp)):

        # dim loop
        coef = 1
        for i, dim in enumerate(dims_interp):

            # bb is the ith bit in b (0 or 1)
            bb = ((1 << i) & b) >> i
            x = x_interp[dim]
            if bb:
                # TODO: can we use += here ?
                coef = coef * x
            else:
                coef = coef * (1 - x)

            keys[data.dims.index(dim)] = iinf[dim] + bb

        result += coef * data_values[tuple(keys)]

    # determine output coords
    coords = {}
    for dim in out_dims:
        if dim in data.coords:
            coords[dim] = data.coords[dim]
        elif dim in ds.coords:
            coords[dim] = ds.coords[dim]

    ret = xr.DataArray(
        result,
        dims=out_dims,
        coords=coords,
    )

    return ret


def selinterp(
    da: xr.DataArray,
    *,
    method: Literal["interp", "nearest"],
    template: Optional[xr.DataArray] = None,
    **kwargs: xr.DataArray,
) -> xr.DataArray:
    """
    Interpolation (or selection) of xarray DataArray `da` along dimensions
    provided as kwargs.

    xarray's `interp` and `sel` methods do not work efficiently on dask-based
    coordinates arrays (full arrays are computed); this function is
    designed to work around this.

    `method`:
        "interp" to apply da.interp
        "nearest" to apply da.sel with method="nearest"

    `template`: the DataArray template for the output of this function.
    By default, use first kwarg.

    Example:
        selinterp(auxdata,
                  method='interp',
                  lat=ds.latitude,
                  lon=ds.longitude)

    This interpolates `auxdata` along dimensions 'lat' and 'lon', by chunks,
    with values defined in ds.latitude and ds.longitude.

    Caveats:
        - Returns float64
        - Cannot specify options per dimension
    """
    warnings.warn(
        "Deprecated function: use function `interp` instead", DeprecationWarning
    )

    # Check that aux is not dask-based
    assert (da.chunks is None) or (
        len(da.chunks) == 0
    ), "Input DataArray should not be dask-based"

    first_dim = list(kwargs.values())[0]
    template = template or first_dim.reset_coords(drop=True).rename(da.name).astype(
        {
            "interp": "float64",
            "nearest": da.dtype,
        }[method]
    )

    # update the template with input attributes
    template.attrs = da.attrs

    func = {
        "interp": interp_chunk,
        "nearest": sel_chunk,
    }[method]

    interpolated = xr.map_blocks(
        func,
        xr.Dataset(kwargs),  # use dim names from source Dataarray
        template=template,
        kwargs={"aux": da},
    )

    return interpolated


def interp_chunk(ds, aux):
    """
    Apply da.interp to a single chunk
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return aux.interp({k: ds[k] for k in ds}).reset_coords(drop=True)


def sel_chunk(ds, aux):
    """
    da.sel of a single chunk
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return aux.sel({k: ds[k] for k in ds}, method="nearest").reset_coords(drop=True)


def interp_legacy(
    aux: xr.DataArray,
    ds_coords: xr.Dataset,
    dims: dict,
    template: Optional[xr.DataArray] = None,
) -> xr.DataArray:
    """
    Interpolation of xarray DataArray `aux` along dimensions provided
    in dask-based Dataset `ds_coords`. The mapping of these dimensions is
    defined in `dims`.

    The xarray `interp` method does not work efficiently on dask-based
    coordinates arrays (full arrays are computed); this function is
    designed to work around this.

    The mapping of variable names is provided in `dims`.

    `template`: the DataArray template for the output of this function.
    By default, use ds_coords[dims[0]].

    Example:
        interp(auxdata, ds, {'lat': 'latitude', 'lon': 'longitude'})

    This interpolates `auxdata` along dimensions 'lat' and 'lon', by chunks,
    with values defined in ds['latitude'] and ds['longitude'].
    """
    warnings.warn(
        "Deprecated function: use function `interp` instead", DeprecationWarning
    )

    def interp_chunk(ds):
        """
        Interpolation of a single chunk
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return aux.interp({k: ds[v] for (k, v) in dims.items()}).reset_coords(
                drop=True
            )

    # Check that aux is not dask-based
    assert (aux.chunks is None) or (
        len(aux.chunks) == 0
    ), "Auxiliary DataArray should not be dask-based"

    first_dim = list(dims.values())[0]
    template = template or ds_coords[first_dim].reset_coords(drop=True)

    # update the interpolated with input attributes
    template.attrs = aux.attrs

    interpolated = xr.map_blocks(
        interp_chunk, ds_coords[dims.keys()], template=template
    )

    return interpolated


def in_index(ind: pd.Index, values: np.ndarray):
    """
    Returns whether each value is within the range defined by `ind`
    """
    if ind.is_monotonic_increasing:
        vmin = ind[0]
        vmax = ind[-1]

    elif ind.is_monotonic_decreasing:
        vmin = ind[-1]
        vmax = ind[0]

    else:
        raise ValueError(
            f"Index of dimension {ind.name} should be either "
            "monotonically increasing or decreasing."
        )

    return (values >= vmin) & (values <= vmax)
