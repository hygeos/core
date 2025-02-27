#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Various utility functions for modifying xarray object
'''

from functools import wraps
import re
from pathlib import Path
from typing import Callable, Union, overload, Dict, List, Tuple
import xarray as xr
import numpy as np
from dask import array as da

from numpy import arcsin as asin
from numpy import cos, radians, sin, sqrt, where
try:
    from shapely.geometry import Point, Polygon
except ImportError:
    pass
from collections import OrderedDict
from dateutil.parser import parse

from core.naming import names


flags          = 'flags'
flags_dtype    = 'uint16'
flags_meanings = 'flag_meanings'
flags_masks    = 'flag_masks'
flags_meanings_separator = ' '

footprint_lat = 'footprint_lat'
footprint_lon = 'footprint_lon'

def datetime(ds: xr.Dataset):
    '''
    Parse datetime (in isoformat) from `ds` attributes
    '''
    if ('start_time' in ds.attrs) and ('end_time' in ds.attrs):
        st = ds.start_time
        et = ds.end_time
        return st + (et - st)/2
    elif 'datetime' in ds.attrs:
        return parse(ds.attrs['datetime']).replace(tzinfo=None)
    else:
        raise AttributeError


def haversine(lat1: float, lon1: float, lat2: float, lon2: float, 
              radius: float = 6371):
    '''
    Calculate the great circle distance between two points (specified in
    decimal degrees) on a sphere of a given radius

    Returns the distance in the same unit as radius (defaults to earth radius in km)
    '''
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = [radians(x) for x in [lon1, lat1, lon2, lat2]]

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    dist = radius * c

    return dist


def locate(lat, lon, lat0, lon0,
           dist_min_km: float = None,
           verbose: bool = False):
    """
    Locate `lat0`, `lon0` within `lat`, `lon`

    if dist_min_km is specified and if the minimal distance
    exceeds it, a ValueError is raised
    """
    if verbose:
        print(f'Locating lat={lat0}, lon={lon0}')
    dist = haversine(lat, lon, lat0, lon0)
    dist_min = np.array(np.nanmin(dist))

    if np.isnan(dist_min):
        raise ValueError('No valid input coordinate')

    if (dist_min_km is not None) and (dist_min > dist_min_km):
        raise ValueError(f'locate: minimal distance is {dist_min}, '
                         f'should be at most {dist_min_km}')

    return [x[0] for x in np.where(dist == dist_min)]


def contains(ds: xr.Dataset, lat: float, lon: float):
    pt = Point(lat, lon)
    area = Polygon(zip(
        ds.attrs[footprint_lat],
        ds.attrs[footprint_lon]
    ))
    # TODO: proper inclusion test
    # TODO: make it work with arrays
    return area.contains(pt)


def sub(ds: xr.Dataset, 
        cond: xr.DataArray, 
        drop_invalid: bool = True, 
        int_default_value: int = 0):
    '''
    Creates a Dataset based on the conditions passed in parameters

    cond : a DataArray of booleans that defines which pixels are kept

    drop_invalid, bool
        if True invalid pixels will be replace by nan for floats and
        int_default_value for other types

    int_default_value, int
        for DataArrays of type int, this value is assigned on non-valid pixels
    '''
    res = xr.Dataset()

    if drop_invalid:
        assert 'mask_valid' not in res
        res['mask_valid'] = cond.where(cond, drop=True)
        res['mask_valid'] = res['mask_valid'].where(~np.isnan(res['mask_valid']), 0).astype(bool)

    slice_dict = dict()
    for dim in cond.dims:
        s = cond.any(dim=[d for d in cond.dims if d != dim])
        wh = where(s)[0]
        if len(wh) == 0:
            slice_dict[dim] = slice(2,1)
        else:
            slice_dict[dim] = slice(wh[0], wh[-1]+1)

    for var in ds.variables:
        if set(cond.dims) == set(ds[var].dims).intersection(set(cond.dims)):
            if drop_invalid:
                if ds[var].dtype in ['float16', 'float32', 'float64']:
                    res[var] = ds[var].where(cond, drop=True)
                else:
                    res[var] = ds[var].isel(slice_dict).where(res['mask_valid'], int_default_value)

            else:
                res[var] = ds[var].isel(slice_dict)

    res.attrs.update(ds.attrs)

    return res


def sub_rect(ds: xr.Dataset, lat_min, lon_min, lat_max, lon_max, 
             drop_invalid: bool = True, int_default_value: int = 0):
    '''
    Returns a Dataset based on the coordinates of the rectangle passed in parameters

    lat_min, lat_max, lon_min, lon_max : delimitations of the region of interest

    drop_invalid, bool : if True, invalid pixels will be replace by nan
    for floats and int_default_value for other types

    int_default_value, int : for DataArrays of type int, this value is assigned on non-valid pixels
    '''
    lat = ds.latitude.compute()
    lon = ds.longitude.compute()
    cond = (lat < lat_max) & (lat > lat_min) & (lon < lon_max) & (lon > lon_min)
    cond = cond.compute()

    return sub(ds, cond, drop_invalid, int_default_value)


def sub_pt(ds: xr.Dataset, pt_lat, pt_lon, rad, 
           drop_invalid: bool = True, int_default_value: int = 0):
    '''
    Creates a Dataset based on the circle specified in parameters

    pt_lat, pt_lon : Coordonates of the center of the point

    rad : radius of the circle in km

    drop_invalid, bool
        if True invalid pixels will be replace by nan for floats
        and int_default_value for other types

    int_default_value, int
        for DataArrays of type int, this value is assigned on non-valid pixels
    '''
    lat = ds[str(names.latitude)].compute()
    lon = ds[str(names.longitude)].compute()
    cond = haversine(lat, lon, pt_lat, pt_lon) < rad
    cond = cond.compute()

    return sub(ds, cond, drop_invalid, int_default_value)


def split(d: xr.Dataset | xr.DataArray, dim: str, sep: str = '_'):
    '''
    Returns a Dataset where a given dimension is split into as many variables

    d: Dataset or DataArray
    '''
    assert dim in d.dims
    assert dim in d.coords, f'The split dimension "{dim}" must have coordinates.'

    if isinstance(d, xr.DataArray):
        m = xr.merge([
            d.isel(**{dim: i}).rename(f'{d.name}{sep}{d[dim].data[i]}').drop_vars(dim)
            for i in range(len(d[dim]))
            ])
    elif isinstance(d, xr.Dataset):
        m = xr.merge(
            [split(d[x], dim)
             if dim in d[x].dims
             else d[x]
             for x in d])
    else:
        raise Exception('`split` expects Dataset or DataArray.')

    m.attrs.update(d.attrs)
    m.attrs['split_dimension'] = dim
    m = m.assign_coords(**d.coords)
    return m


def merge(ds: xr.Dataset,
          dim: str = None,
          varname: str = None,
          pattern: str = r'(.+)_(\d+)',
          dtype: type = int):
    r"""
    Merge DataArrays in `ds` along dimension `dim`.

    ds: xr.Dataset

    dim: str or None
        name of the new or existing dimension
        if None, use the attribute `split_dimension`
    
    varname: str or None
        name of the variable to create
        if None, detect variable name from regular expression

    pattern: str
        Regular expression for matching variable names and coordinates
        if varname is None:
            First group represents the new variable name.
            Second group represents the coordinate value
            Ex: r'(.+)_(\d+)'
                    First group matches all characters.
                    Second group matches digits.
                r'(\D+)(\d+)'
                    First group matches non-digit.
                    Second group matches digits.
        if varname is not None:
            Match a single group representing the coordinate value

    dtype: data type
        data type of the coordinate items
    """
    copy = ds.copy()

    if dim is None:
        dim = copy.attrs['split_dimension']

    mapping = {}   # {new_name: [(old_name, value), ...], ...}
    for x in copy:
        m = re.findall(pattern, x)
        if not m:
            continue  # does not match
        assert len(m) == 1, 'Expecting a single match'

        if varname is None:
            assert len(m[0]) == 2, 'Expecting two groups in regular expression'
            var, coord = m[0]
        else:
            assert not isinstance(m[0], tuple), 'Expecting a single group in regular expression'
            coord = m[0]
            var = varname
        c = dtype(coord)

        if var not in mapping:
            mapping[var] = []

        mapping[var].append((x, c))

    for var in mapping:
        data = xr.concat([copy[x] for x, c in mapping[var]], dim)
        coords = [c for x, c in mapping[var]]
        if dim in copy.coords:
            # check that the coordinates are matching
            existing_coords = list(copy.coords[dim].data)
            assert existing_coords == coords, \
                f'Error: {existing_coords} != {coords} (in variable {var})'
        else:
            copy = copy.assign_coords(**{dim: coords})
        copy[var] = data
        copy = copy.drop_vars([x for x, c in mapping[var]])

    return copy


def getflags(A=None, meanings=None, masks=None, sep=None):
    """
    returns the flags in attributes of `A` as a dictionary {meaning: value}

    Arguments:
    ---------

    provide either:
        A: Dataarray
    or:
        meanings: flag meanings 'FLAG1 FLAG2'
        masks: flag values [1, 2]
        sep: string separator
    """
    try:
        meanings = meanings if (meanings is not None) else A.attrs[flags_meanings]
        masks = masks if (masks is not None) else A.attrs[flags_masks]
        sep = sep or flags_meanings_separator
    except KeyError:
        return OrderedDict()
    return OrderedDict(zip(meanings.split(sep), masks))


def getflag(A: xr.DataArray, name: str):
    """
    Return the binary flag with given `name` as a boolean array

    A: DataArray
    name: str

    example: getflag(flags, 'LAND')
    """
    flags = getflags(A)

    assert name in flags, f'Error, {name} no in {list(flags)}'

    return (A & flags[name]) != 0


def raiseflag(A: xr.DataArray, flag_name: str, flag_value: int, condition=None):
    """
    Raise a flag in DataArray `A` with name `flag_name`, value `flag_value` and `condition`
    The name and value of the flag is recorded in the attributes of `A`

    Arguments:
    ----------
    A: DataArray of integers

    flag_name: str
        Name of the flag
    flag_value: int
        Value of the flag
    condition: boolean array-like of same shape as `A`
        Condition to raise flag.
        If None, the flag values are unchanged ; the flag is simple registered in the
        attributes.
    """
    flags = getflags(A)
    dtype_flag_masks = 'uint16'

    if flags_meanings not in A.attrs:
        A.attrs[flags_meanings] = ''
    if flags_masks not in A.attrs:
        A.attrs[flags_masks] = np.array([], dtype=dtype_flag_masks)

    # update the attributes if necessary
    if flag_name in flags:
        # existing flag: check value
        assert flags[flag_name] == flag_value, \
            f'Flag {flag_name} already exists with a different value'
    else:
        assert flag_value not in flags.values(), \
            f'Flag value {flag_value} is already assigned to a different flags (assigned flags are {flags.values()})'

        flags[flag_name] = flag_value

        # sort the flags by values
        keys, values = zip(*sorted(flags.items(), key=lambda y: y[1]))

        A.attrs[flags_meanings] = flags_meanings_separator.join(keys)
        A.attrs[flags_masks] = np.array(values, dtype=dtype_flag_masks)

    if condition is not None:
        notraised = (A & flag_value) == 0
        A += flag_value * ((condition != 0) & notraised).astype(flags_dtype)


def wrap(ds: xr.Dataset, dim: str, vmin: float, vmax: float):
    """
    Wrap and reorder a cyclic dimension between vmin and vmax.
    The border value is duplicated at the edges.
    The period is (vmax-vmin)

    Example:
    * Dimension [0, 359] -> [-180, 180]
    * Dimension [-180, 179] -> [-180, 180]
    * Dimension [0, 359] -> [0, 360]

    Arguments:
    ----------

    ds: xarray.Dataset
    dim: str
        Name of the dimension to wrap
    vmin, vmax: float
        new values for the edges
    """

    pivot = vmax if (vmin < ds[dim][0]) else vmin

    left = ds.sel({dim: slice(None, pivot)})
    right = ds.sel({dim: slice(pivot, None)})

    if right[dim][-1] > vmax:
        # apply the offset at the right part
        right = right.assign_coords({dim: right[dim] - (vmax-vmin)})
    else:
        # apply the offset at the left part
        left = left.assign_coords({dim: left[dim] + (vmax-vmin)})

    # swaps the two parts
    return xr.concat([right, left], dim=dim)


def convert(A: xr.DataArray, unit_to: str, unit_from: str = None, converter: dict = None):
    """
    Unit conversion

    Arguments:
    ---------

    A: DataArray to convert

    unit_from: str or None
        unit to convert from. If not provided, uses da.units

    unit_to: str
        unit to convert to
    
    converter: a dictionary for unit conversion
        example: converter={'Pa': 1, 'hPa': 1e-2}
    """
    if unit_from is None:
        unit_from = A.units
    
    default_converters = [
        # pressure
        {'Pa': 1,
         'hPa': 1e-2,
         'millibars': 1e-2,
         },

        # ozone
        {'kg/m2': 1,
         'kg m**-2': 1,
         'DU': 1/2.1415E-05,
         'Dobson units': 1/2.1415E-05,
         }
    ]

    conversion_factor = None
    for c in (default_converters if converter is None else [converter]):
        if (unit_from in c) and (unit_to in c):
            conversion_factor = c[unit_to]/c[unit_from]
            break

    if conversion_factor is None:
        raise ValueError(f'Unknown conversion from {unit_from} to {unit_to}')

    converted = A*conversion_factor
    converted.attrs['units'] = unit_to
    return converted


def chunk(ds: xr.Dataset, **kwargs):
    """
    Apply rechunking to a xr.Dataset `ds` along dimensions provided as kwargs

    Works like `ds.chunk` but works also for Datasets with repeated dimensions.
    """

    for var in ds:
        chks = [kwargs[d] if d in kwargs else None for d in ds[var].dims]
        if hasattr(ds[var].data, 'chunks') and len([c for c in chks if c is not None]):
            ds[var].data = ds[var].data.rechunk(chks)
            
    return ds


def trim_dims(A: xr.Dataset):
    """
    Trim the dimensions of Dataset A
    
    Rename all possible dimensions to avoid duplicate dimensions with same sizes
    Avoid any DataArray with duplicate dimensions
    """
    # list of lists of dimensions that should be grouped together
    groups = []
    
    # loop over all dimensions sizes
    for size in set(A.dims.values()):
        # list all dimensions with current size
        groups_current = []
        dims_current = [k for k, v in A.dims.items()
                        if v == size]

        # for each variable, add its dimensions (intersecting dims_current)
        # to separate groups to avoid duplicated
        for var in A:
            for i, d in enumerate(
                [x for x in A[var].dims
                 if x in dims_current]
                ):
                if len(groups_current) <= i:
                    groups_current.append([])
                if d not in groups_current[i]:
                    groups_current[i].append(d)

        groups += groups_current

    # check that intersection of all groups is empty
    assert not set.intersection(*[set(x) for x in groups])

    rename_dict = dict(sum([[(dim, 'new_'+group[0])
                             for dim in group]
                            for group in groups
                            if len(group) > 1  # don't rename if not useful
                            ], []))
    return A.rename_dims(rename_dict)


def only(iterable):
    """If *iterable* has only one item, return it.
    Otherwise raise a ValueError
    """
    x = list(iterable)
    if len(x) != 1:
        raise ValueError
    return x[0] 

def reglob(path: Path|str, regexp: str):
    files = []
    assert regexp[0] != '*', 'Avoid using wildcard in regular expression'
    for p in Path(path).iterdir(): 
        a = re.search(regexp, str(p))
        if a is not None:
            files.append(a.group())
    return files

@overload
def xrcrop(A: xr.Dataset, **kwargs) -> xr.Dataset: ...
@overload
def xrcrop(A: xr.DataArray, **kwargs) -> xr.DataArray: ...


def xrcrop(
    A: Union[xr.Dataset, xr.DataArray], **kwargs
) -> Union[xr.Dataset, xr.DataArray]:
    """
    Crop a Dataset or DataArray along dimensions based on min/max values.
    
    For each dimension provided as kwarg, the min/max values along that dimension
    can be provided:
        - As a min/max tuple
        - As a DataArrat, for which the min/max are computed

    Ex: crop dimensions `latitude` and `longitude` of `gsw` based on the min/max
        of ds.lat and ds.lon
        gsw = xrcrop(
            gsw,
            latitude=ds.lat,
            longitude=ds.lon,
        )
    
    Note: the purpose of this function is to make it possible to .compute() the result
    of the cropped data, thus allowing to perform a sel over large arrays (otherwise
    extremely slow with dask based arrays).
    """
    isel_dict = {}
    for k, v in kwargs.items():
        if isinstance(v, tuple):
            (vmin, vmax) = v
            assert vmin < vmax
        elif isinstance(v, xr.DataArray):
            vmin = v.min().compute().item()
            vmax = v.max().compute().item()
        else:
            raise TypeError
        index = A.indexes[k]

        # Get bracketing indices for vmin/vmax
        if index.is_monotonic_increasing:
            imin = max(0, index.get_slice_bound(vmin, "right") - 1)
            imax = min(len(index), index.get_slice_bound(vmax, "left") + 1)
        elif index.is_monotonic_decreasing:
            imin = max(0, index.get_slice_bound(vmax, "right") - 1)
            imax = min(len(index), index.get_slice_bound(vmin, "left") + 1)
        else:
            raise ValueError

        assert imin < imax
        isel_dict[k] = slice(imin, imax)

    return A.isel(isel_dict)


class MapBlocksOutput:
    def __init__(
        self,
        model: List,
        new_dims: Dict | None = None,
    ) -> None:
        """
        Describe a Dataset structure, for use with xr.map_blocks to ensure consistency
        between the output of the blockwise function and the call to xr.map_blocks.

        Args:
            model: list of DataArrays of Var objects describing the output
            new_dims: dictionary providing the new dimensions
                ex: new_dims={'new_dim': xr.DataArray([0, 2, 3])}
                or simply the dimension size if new_dim has no coordinate

        Example:
            model = MapBlockOutput([
                # Either describe the variable with a `Var` object
                Var('latitude', 'float32', ['x', 'y']),
                # or with a renamed DataArray
                ds.rho_toa.rename('rho_gc')
            ])

        """
        self.model = model
        self.new_dims = new_dims or {}

    def __add__(self, other):
        concatenated = MapBlocksOutput(self.model + other.model,
                               new_dims={**self.new_dims, **other.new_dims})
        names = [x.name for x in concatenated.model]
        assert len(set(names)) == len(names)
        return concatenated

    def template(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Return an empty template for this model, to be provided to xr.map_blocks
        """
        return xr.merge(
            [
                m.to_dataarray(ds, self.new_dims) if isinstance(m, Var) else m
                for m in self.model
            ]
        )

    def subset(self, ds: xr.Dataset) -> xr.Dataset:
        return ds[[var.name for var in self.model]]

    def conform(self, ds: xr.Dataset, transpose: bool = False) -> xr.Dataset:
        """
        Conform dataset `ds` to this model

        transpose: whether to automatically transpose the variables in `ds` to conform
            to the specified dimensions.
        """
        list_vars = []
        for var in self.model:
            da = ds[var.name]
            if da.dtype != var.dtype:
                raise TypeError(
                    f'Expected type "{var.dtype}" for "{var.name}" but '
                    f'encountered "{da.dtype}"'
                )

            if da.dims != var.dims:
                if set(da.dims) != set(var.dims):
                    raise RuntimeError(
                        f'Expected dimensions "{var.dims}" for variable "{var.name}" '
                        f'but encountered "{da.dims}".'
                    )
                if transpose:
                    list_vars.append(da.transpose(*var.dims))
                else:
                    raise RuntimeError(
                        f'Expected dimensions "{var.dims}" for variable "{var.name}" '
                        f'but encountered "{da.dims}". Please consider `transpose=True`'
                    )
            else:
                list_vars.append(da)

        return xr.merge(list_vars)


class Var:
    def __init__(self, name: str, dtype: str, dims: Tuple):
        """
        Represents a DataArray structure as used by `MapBlockOutput`
        """
        self.name = name
        self.dtype = dtype
        self.dims = dims

    def to_dataarray(self, ds: xr.Dataset, new_dims: Dict | None = None):
        """
        Convert to a DataArray with dims infos provided by `ds`

        Args:
            new_dims: size of each new dimension
            new_coords: coords of each new dimension
        """
        new_dims = new_dims or {}
        shape = []
        chunks = []
        coords = {}
        for d in self.dims:
            if d in ds.dims:
                shape.append(len(ds[d]))
                chunks.append(ds.chunks[d])
            else:
                if d not in new_dims:
                    raise RuntimeError(f'dimension "{d}" has not been described.')
                if hasattr(new_dims[d], "__len__"):
                    n = len(new_dims[d])
                else:
                    n = new_dims[d]
                shape.append(n)
                chunks.append((n,))

            if d in ds.coords:
                coords[d] = ds.coords[d].values
            elif (d in new_dims) and hasattr(new_dims[d], "__len__"):
                coords[d] = new_dims[d]

        return xr.DataArray(
            da.empty(shape=shape, dtype=self.dtype, chunks=chunks),
            dims=self.dims,
            name=self.name,
            coords=coords,
        )


def xr_filter(
    ds: xr.Dataset,
    condition: xr.DataArray,
    stackdim: str | None = None,
    transparent: bool = False,
) -> xr.Dataset:
    """
    Extracts a subset of the dataset where the condition is True, stacking the
    `condition` dimensions. Equivalent to numpy's boolean indexing, A[condition].

    Parameters:
    ds (xr.Dataset): The input dataset.
    condition (xr.DataArray): A boolean DataArray indicating where the condition is True.
    stackdim (str, optional): The name of the new stacked dimension. If None, it will be
        determined automatically from the condition dimensions.
    transparent (bool, optional): whether to reassign the original dimension names to
        the Dataset (expanding with length-one dimensions).

    Returns:
    xr.Dataset: A new dataset with the subset of data where the condition is True.
    """
    stackdim = stackdim or "_".join(condition.dims)
    assert stackdim not in ds.dims
    ok = condition.stack({stackdim: condition.dims})

    # Extract sub Dataset
    stacked = ds.stack({stackdim: condition.dims})
    sub = xr.Dataset()
    for var in stacked:
        da = stacked[var]
        sub[var] = xr.DataArray(
            da.values[
                tuple(ok if (dim == stackdim) else slice(None) for dim in da.dims)
            ],
            dims=da.dims,
        )

    # Assign required coordinates to sub
    sub = sub.assign_coords(
        {dim: ds.coords[dim] for dim in ds.coords if dim not in condition.coords}
    )
    
    if transparent:
        # reassign the initial dimension names to the Dataset
        sub = sub.rename({stackdim: condition.dims[0]}).expand_dims(*condition.dims[1:])
    return sub


def xr_unfilter(
    sub: xr.Dataset,
    condition: xr.DataArray,
    stackdim: str | None = None,
    fill_value_float: float = np.nan,
    fill_value_int: int = 0,
    transparent: bool = False,
) -> xr.DataArray:
    """
    Reconstructs the original dataset from a subset dataset where the condition is True,
    unstacking the condition dimensions.

    Parameters:
    sub (xr.Dataset): The subset dataset where the condition is True.
    condition (xr.DataArray): A boolean DataArray indicating where the condition is True.
    stackdim (str, optional): The name of the stacked dimension. If None, it will be
        determined automatically from the condition dimensions.
    fill_value_float (float, optional): The fill value for floating point data types.
        Default is np.nan.
    fill_value_int (int, optional): The fill value for integer data types. Default is 0.
    transparent (bool, optional): whether to revert the transparent compatibility
        conversion applied in xrwhere.

    Returns:
    xr.DataArray: The reconstructed dataset with the specified dimensions unstacked.
    """
    stackdim = stackdim or "_".join(condition.dims)

    if transparent:
        sub = sub.rename({condition.dims[0]: stackdim}).squeeze([*condition.dims[1:]])

    assert stackdim in sub.dims
    ok = condition.stack({stackdim: condition.dims})

    stacked = xr.Dataset()
    for var in sub:
        # determine the shape of the stacked array
        stacked_shape = tuple(
            sub[dim].size if dim != stackdim else condition.size
            for dim in sub[var].dims
        )

        # determine fill value
        dtype = sub[var].dtype
        if np.issubdtype(dtype, np.floating):
            fill_value = fill_value_float
        elif np.issubdtype(dtype, np.integer):
            fill_value = fill_value_int
        else:
            raise ValueError(f"Undefined fill value for type {dtype}")

        # initialize the stacked array
        stacked[var] = xr.DataArray(
            np.full(stacked_shape, fill_value, dtype=dtype), dims=sub[var].dims
        )

        # affect the sub values to the stacked array
        stacked[var].values[
            tuple(ok if (dim == stackdim) else slice(None) for dim in sub[var].dims)
        ] = sub[var].values

    # unstack the array
    full = stacked.assign_coords({stackdim: ok[stackdim]}).unstack(stackdim)

    # remove all coords
    full = full.drop_vars(full.coords)

    # reassign all required coords
    full = full.assign_coords(sub.coords).assign_coords(condition.coords)

    return full


def xr_filter_decorator(
    argpos: int,
    condition: Callable,
    fill_value_float: float = np.nan,
    fill_value_int: int = 0,
    transparent: bool = False,
    stackdim: str | None = None,
):
    """
    A decorator which applies the decorated function only where the condition is True.

    Args:
        argpos (int): Position index of the input dataset in the decorated function call.
        condition (Callable): A callable taking the Dataset as input and returning a
            boolean DataArray.
        fill_value_float (float, optional): Fill value for floating point data types.
            Default is np.nan.
        fill_value_int (int, optional): Fill value for integer data types.
            Default is 0
        transparent (bool, optional): Whether to reassign the original dimension names
            to the Dataset (expanding with length-one dimensions). Default is False.
        stackdim (str | None, optional): The name of the new stacked dimension.
            If None, it will be determined automatically from the condition dimensions. Default is None.

    Example:
        @xr_filter_decorator(0, lambda x: x.flags == 0)
        def my_func(ds: xr.Dataset) -> xr.Dataset:
            # my_func is applied only where ds.flags == 0
            ...

    The decorator works by:
    1. Extracting a subset of the dataset where the condition is True using `xr_filter`.
    2. Applying the decorated function to the subset.
    3. Reconstructing the original dataset from the subset using `xr_unfilter`.

    NOTE: this decorator does not guarantee that the order of dimensions is maintained.
    When using this decorator with `xr.apply_blocks`, you may want to wrap your
    xr_filter_decorator decorated method with the `conform` decorator.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            ds = args[argpos]
            ok = condition(ds)
            sub = xr_filter(
                ds,
                ok,
                stackdim=stackdim,
                transparent=transparent,
            )
            new_args = tuple(sub if i == argpos else a for i, a in enumerate(args))
            result = func(*new_args, **kwargs)
            return xr_unfilter(
                result,
                ok,
                fill_value_float=fill_value_float,
                fill_value_int=fill_value_int,
                stackdim=stackdim,
                transparent=transparent,
            )

        return wrapper

    return decorator


def conform(attrname: str, transpose: bool = True):
    """
    A method decorator which applies MapBlocksOutput.conform to the method output.

    The MapBlocksOutput should be an attribute `attrname` of the class.
    """

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # get the MapBlocksOutput model
            model = getattr(self, attrname)

            # apply the function
            result = func(self, *args, **kwargs)

            # return the conformed result
            return model.conform(result, transpose=transpose)

        return wrapper

    return decorator