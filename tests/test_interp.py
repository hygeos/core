import numpy as np
import pytest
from eoread.common import timeit
from core.interpolate import (
    Nearest_Indexer,
    interp,
    interp_v1,
    selinterp,
    Linear,
    Nearest,
)
from luts import luts
from core import conftest
import xarray as xr
from core.pytest_utils import parametrize_dict


list_interp_functions = {
    "xarray": lambda data, l1: data.interp(
        lat=l1.latitude,
        lon=l1.longitude,
    ),
    "selinterp": lambda data, l1: selinterp(
        data,
        lat=l1.latitude,
        lon=l1.longitude,
        method="interp",
    ),
    "interp": lambda data, l1: interp(
        # interp_v1
        data,
        interp={
            "lat": l1.latitude,
            "lon": l1.longitude,
        },
    ),
    "lut": lambda data, l1: xr.DataArray(
        luts.from_xarray(data)[
            luts.Idx(l1.latitude.values), luts.Idx(l1.longitude.values)
        ]
    ),
    "interp_v2": lambda data, l1: interp(
        # interp_v2
        data,
        lat=Linear(l1.latitude),
        lon=Linear(l1.longitude),
    ),
    "interp_v2_non_regular": lambda data, l1: interp(
        # interp_v2
        data,
        lat=Linear(l1.latitude, regular='no'),
        lon=Linear(l1.longitude, regular='no'),
    ),
}


@pytest.mark.parametrize("apply_function", **parametrize_dict(list_interp_functions))
@pytest.mark.parametrize("mode", [
    'numpy',
    'dask',
    ])
def test_perf_interp(mode: str, apply_function: callable, request):
    factor = 1 # 1 for fast, 10 for slower
    l1 = xr.Dataset()
    shp = (500*factor, 500*factor)
    l1['latitude'] = xr.DataArray(np.random.random(shp)*20 - 10, dims=['x', 'y'])
    l1['longitude'] = xr.DataArray(np.random.random(shp)*20 - 10, dims=['x', 'y'])
    
    if mode == "dask":
        # numpy mode: pre-compute the array of coordinates
        l1 = l1.chunk({'x':50*factor, 'y': 50*factor})

    # dummy ancillary data
    shp_anc = (180, 360)
    data = xr.DataArray(
        np.random.random(shp_anc),
        dims=["lat", "lon"],
        coords={
            "lat": np.linspace(-90, 90, shp_anc[0]),
            "lon": np.linspace(-180, 180, shp_anc[1]),
        },
    )

    with timeit("INST"):
        interpolated = apply_function(data, l1)

    with timeit("COMP"):
        interpolated.compute()



def sample(vmin: float, vmax: float, dims: list):
    sizes = {
        "x": 10,
        "y": 15,
        "z": 20,
    }
    shp = [sizes[dim] for dim in dims]
    return xr.DataArray(
        vmin + np.random.random(np.prod(shp)).reshape(shp) * (vmax - vmin),
        dims=dims,
        # coords={dim: np.linspace(0, 50, shp[i]) for i, dim in enumerate(dims)},
    ).chunk({d: sizes[d] for d in dims})


@pytest.mark.parametrize(
    "kwargs",
    [
        {  # same dimensions
            "interp": {
                "a": sample(1, 3, ["x", "y"]),
                "b": sample(11, 14, ["x", "y"]),
                "c": sample(101, 105, ["x", "y"]),
            },
            "sel": {},
        },
        {  # mixed dimensions
            "interp": {
                "b": sample(11, 14, ["x", "y"]),
                "c": sample(101, 105, ["x", "z"]),
            },
            "sel": {"a": sample(1, 3, ["z"])},
            "options": {"a": {"method": "nearest"}},
        },
        {  # scalar
            "interp": {
                "b": sample(11, 14, ["x", "y"]),
                "c": sample(101, 105, ["x", "z"]),
            },
            "sel": {"a": 2},
        },
        {  # missing dimension (slice)
            "interp": {
                "b": sample(11, 14, ["x", "y"]),
                "c": sample(101, 105, ["x", "z"]),
            },
        },
        {  # missing dimension (slice)
            "interp": {
                "a": sample(1, 3, ["x", "y"]),
                "c": sample(101, 105, ["x", "z"]),
            },
        },
        {  # missing dimension and reversed dimensions
            "interp": {
                "a": sample(1, 3, ["x", "y"]),
                "c": sample(101, 105, ["y", "x"]),
            },
        },
    ],
)
def test_interp_v1(kwargs):
    data = xr.DataArray(
        np.zeros((3, 4, 5), dtype="float32"),
        dims=["a", "b", "c"],
        coords={
            "a": [1.0, 2.0, 4.0],
            "b": [11.0, 12.0, 13.0, 14.0],
            "c": [105.0, 104.0, 103.0, 102.0, 101.0],
        },
    )
    res = interp_v1(data, **kwargs)

    res.compute()


@pytest.mark.parametrize(
    "kwargs",
    [
        {  # same dimensions
            "a": Linear(sample(1, 3, ["x", "y"])),
            "b": Linear(sample(11, 14, ["x", "y"])),
            "c": Linear(sample(101, 105, ["x", "y"])),
        },
        {  # mixed dimensions
            "a": Nearest(sample(1, 3, ["z"]), tolerance=1.),
            "b": Linear(sample(11, 14, ["x", "y"])),
            "c": Linear(sample(101, 105, ["x", "z"])),
        },
        {  # scalar selection
            "a": Nearest(2),
            "b": Linear(sample(11, 14, ["x", "y"])),
            "c": Linear(sample(101, 105, ["x", "z"])),
        },
        {  # missing dimension (slice)
            "b": Linear(sample(11, 14, ["x", "y"])),
            "c": Linear(sample(101, 105, ["x", "z"])),
        },
        {  # missing dimension (slice)
            "a": Linear(sample(1, 3, ["x", "y"])),
            "c": Linear(sample(101, 105, ["x", "z"])),
        },
        {  # missing dimension and reversed dimensions
            "a": Linear(sample(1, 3, ["x", "y"])),
            "c": Linear(sample(101, 105, ["y", "x"])),
        },
    ],
)
def test_interp_v2(kwargs):
    data = xr.DataArray(
        np.zeros((3, 4, 5), dtype="float32"),
        dims=["a", "b", "c"],
        coords={
            "a": [1.0, 2.0, 4.0],
            "b": [11.0, 12.0, 13.0, 14.0],
            "c": [105.0, 104.0, 103.0, 102.0, 101.0],
        },
    )
    res = interp(data, **kwargs)

    res.compute()

@pytest.mark.parametrize('regular', [True, False])
@pytest.mark.parametrize("interp_version", [1, 2])
def test_decreasing(request, regular, interp_version):
    if regular:
        a = [105.0, 104.0, 103.0, 102.0, 101.0]
    else:
        a = [106.0, 104.0, 103.0, 102.0, 101.0]
    data = xr.DataArray(
        np.array(a),
        dims=["a"],
        coords={
            "a": a,
        },
    )
    if interp_version == 1:
        assert interp_v1(data, interp={"a": xr.DataArray(104.2)}) == 104.2
        interpolated = interp_v1(
            data,
            interp={"a": xr.DataArray(np.linspace(90, 110, 200), dims=["x"])},
            options={"a": {"bounds": "clip"}},
        )
    else:
        # interp_v2
        assert interp(data, a = Linear(xr.DataArray(104.2))) == 104.2
        interpolated = interp(
            data,
            a = Linear(xr.DataArray(np.linspace(90, 110, 200), dims=["x"]), bounds="clip"),
        )

    # Plot some interpolated data
    interpolated.plot()
    conftest.savefig(request)


@pytest.fixture(params=[1, 2])
def fixed_sample(request) -> xr.DataArray:
    A = np.arange(10)
    if request.param == 2:
        # irregular grid
        A[1] += 0.1
    return xr.DataArray(
        np.eye(10),
        dims=["a", "b"],
        coords={
            "a": A,
            "b": A,
        },
    )


@pytest.mark.parametrize("interp_version", [1, 2])
def test_oob_sel(fixed_sample, interp_version):
    # Selection failing because of out of bounds
    with pytest.raises(ValueError):
        if interp_version == 1:
            interp_v1(
                fixed_sample,
                sel={
                    "b": xr.DataArray([-2]),
                },
            )
        else:
            interp(
                fixed_sample,
                b = Nearest(xr.DataArray([-2])),
            )


@pytest.mark.parametrize("interp_version", [1, 2])
def test_oob_sel_nearest(fixed_sample, interp_version):
    # This should pass
    if interp_version == 1:
        interp_v1(
            fixed_sample,
            sel={
                "b": xr.DataArray([-2]),
            },
            options={"b": {"method": "nearest"}},
        )
    else:
        interp(
            fixed_sample,
            b = Nearest(xr.DataArray([-2]), tolerance=None)
        )


@pytest.mark.parametrize("interp_version", **parametrize_dict({
    'interp_v1': 1,
    'interp_v2': 2,
}))
def test_oob_interp(fixed_sample, interp_version):
    # Interpolation failing because of out of bounds
    with pytest.raises(ValueError):
        if interp_version == 1:
            interp_v1(
                fixed_sample,
                interp={
                    "b": xr.DataArray([-2]),
                },
            )
        else:
            interp(
                fixed_sample,
                b = Linear(xr.DataArray([-2]))
            )
    with pytest.raises(ValueError):
        if interp_version == 1:
            interp_v1(
                fixed_sample,
                interp={
                    "b": xr.DataArray([10]),
                },
            )
        else:
            interp(
                fixed_sample,
                b = Linear(xr.DataArray([10])),
            )

    if interp_version == 1:
        interp_v1(
            fixed_sample,
            interp={
                "b": xr.DataArray([9]),
            },
        )
    else:
        interp(
            fixed_sample,
            b = Linear(xr.DataArray([9])),
        )


@pytest.mark.parametrize("interp_version", [1, 2])
def test_oob_interp_clip(fixed_sample, interp_version):
    # Passes because clip=True
    if interp_version == 1:
        interp_v1(
            fixed_sample,
            interp={
                "b": xr.DataArray([-2]),
            },
            options={
                "b": {"bounds": "clip"},
            },
        )
    else:
        interp(
            fixed_sample,
            b=Linear(xr.DataArray([-2]), bounds="clip"),
        )



@pytest.mark.parametrize("A", **parametrize_dict({
    'float_array': np.array([1., 2., 4.]), 
    'int_array': np.array([1, 2, 4]), 
    'int_decreasing': np.array([1, 2, 4])[::-1], 
}))
def test_nearest_indexer(A):
    
    # basic exact indexing
    indexer = Nearest_Indexer(A, 1e-8)
    assert (indexer(A)[0][0] == np.array([0, 1, 2])).all()

    for value in [0, 1.5 , 5.]:
        with pytest.raises(ValueError):
            indexer(value)

    # inexact indexing
    indexer = Nearest_Indexer(A, 0.1)
    assert (indexer(A+0.05)[0][0] == np.array([0, 1, 2])).all()
    assert (indexer(A-0.05)[0][0] == np.array([0, 1, 2])).all()
    
