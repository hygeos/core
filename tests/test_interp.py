try:
    from matplotlib import pyplot as plt
except:
    pass
import numpy as np
import pytest


from core.monitor import Chrono


from core.interpolate import (
    Linear_Indexer,
    Nearest_Indexer,
    Spline_Indexer,
    interp,
    Linear,
    Nearest,
    Spline,
)
try:
    from luts import luts 
except:
    pass

from core import conftest
import xarray as xr
from core.pytest_utils import parametrize_dict


list_interp_functions = {
    "xarray": lambda data, l1: data.interp(
        lat=l1.latitude,
        lon=l1.longitude,
    ),
    # "lut": lambda data, l1: xr.DataArray(
    #     luts.from_xarray(data)[
    #         luts.Idx(l1.latitude.values), luts.Idx(l1.longitude.values)
    #     ]
    # ),
    "interp": lambda data, l1: interp(
        data,
        lat=Linear(l1.latitude),
        lon=Linear(l1.longitude),
    ),
    "interp_non_regular": lambda data, l1: interp(
        data,
        lat=Linear(l1.latitude, spacing='irregular'),
        lon=Linear(l1.longitude, spacing='irregular'),
    ),
}


@pytest.mark.parametrize("reverse", **parametrize_dict({
    "increasing": False,
    "decreasing": True,
}))
@pytest.mark.parametrize("coords", **parametrize_dict({
    'regular': np.linspace(10, 15, 6),
    'irregular': np.array([10., 11., 12., 13., 15.]),
}))
@pytest.mark.parametrize("values,oob", **parametrize_dict({
    "scalar": (np.array(11.2), False),
    "scalar-edge1": (np.array(10.), False),
    "scalar-edge2": (np.array(15.), False),
    "scalar-oob": (np.array(9.), True),
    "array": (np.array([np.nan, 9., 10., 10.5, 14.5, 15., 16.]), True),
    "array2D": (np.array([[np.nan, 9., 10.], [14.5, 15., 16.]]), True),
}))
@pytest.mark.parametrize("indexer_factory", **parametrize_dict({ 
    'nearest': lambda c: Nearest_Indexer(c, tolerance=3),
    'linear_nan': lambda c: Linear_Indexer(c, bounds="nan", spacing="auto"),
    'linear_error': lambda c: Linear_Indexer(c, bounds="error", spacing="auto"),
    'linear_clip': lambda c: Linear_Indexer(c, bounds="clip", spacing="auto"),
    'linear_cycle': lambda c: Linear_Indexer(c, bounds="cycle", spacing="auto"),
    'spline_nan': lambda c: Spline_Indexer(c, bounds="nan", spacing="auto", tension=0.5),
    'spline_error': lambda c: Spline_Indexer(c, bounds="error", spacing="auto", tension=0.5),
    'spline_clip': lambda c: Spline_Indexer(c, bounds="clip", spacing="auto", tension=0.5),
}))
def test_indexer(indexer_factory, coords, values, oob, reverse):
    """
    Thoroughly test the indexers
    """
    if reverse:
        coords = coords[::-1]

    # instantiate the indexer
    try:
        indexer = indexer_factory(coords)
    except RuntimeError:
        # in case of bounds="cycle" and irregular cooefs
        return
    
    if hasattr(indexer, 'bounds') and (indexer.bounds == "error") and oob:
        with pytest.raises(ValueError):
            indexer(values)
    else:
        for idx, w in indexer(values):
            assert (idx >= 0).all()
            assert (idx < len(coords)).all()
            if (w is not None) and not isinstance(indexer, Spline_Indexer):
                w = np.array(w)
                ok = ~np.isnan(w)
                assert (w[ok] >= 0).all()
                assert (w[ok] <= 1).all()

        # check bracketing values (for scalars - linear only)
        if (not oob) and (values.ndim == 0) and isinstance(indexer, Linear_Indexer):
            i1, w1 = indexer(values)[0]
            i2, w2 = indexer(values)[1]
            v1 = coords[i1]
            v2 = coords[i2]
            assert np.isclose(values, v1*w1 + v2*w2)
            if indexer.bounds != "cycle":
                assert v1 <= values
                assert v2 >= values


def test_indexer_time():
    indexer = Linear_Indexer(
        np.datetime64('2025-01-01') + np.arange(10)*np.timedelta64(1, 'D'),
        bounds="error",
        spacing="auto",
    )
    value = np.array(np.datetime64('2025-01-02 10:30'))
    indexer(value)


def test_cycle_period():
    coords = np.linspace(-180, 179, 360)
    Linear_Indexer(coords, bounds="cycle", spacing="auto", period=360),
    with pytest.raises(AssertionError):
        Linear_Indexer(coords, bounds="cycle", spacing="auto", period=359),


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

    with Chrono("INST"):
        interpolated = apply_function(data, l1)

    with Chrono("COMP"):
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
        {  # same dimensions
            "a": Spline(sample(1, 3, ["x", "y"])),
            "b": Spline(sample(11, 14, ["x", "y"])),
            "c": Spline(sample(101, 105, ["x", "y"])),
        },
        {  # mixed dimensions
            "a": Nearest(sample(1, 3, ["z"]), tolerance=1.),
            "b": Spline(sample(11, 14, ["x", "y"])),
            "c": Spline(sample(101, 105, ["x", "z"])),
        },
        {  # scalar selection
            "a": Nearest(2),
            "b": Spline(sample(11, 14, ["x", "y"])),
            "c": Spline(sample(101, 105, ["x", "z"])),
        },
        {  # missing dimension (slice)
            "b": Spline(sample(11, 14, ["x", "y"])),
            "c": Spline(sample(101, 105, ["x", "z"])),
        },
        {  # missing dimension (slice)
            "a": Spline(sample(1, 3, ["x", "y"])),
            "c": Spline(sample(101, 105, ["x", "z"])),
        },
        {  # missing dimension and reversed dimensions
            "a": Spline(sample(1, 3, ["x", "y"])),
            "c": Spline(sample(101, 105, ["y", "x"])),
        },
    ],
)
def test_interp(kwargs):
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
def test_decreasing(request, regular):
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
    assert np.isclose(interp(data, a = Linear(xr.DataArray(104.2))),104.2)
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


def test_oob_sel(fixed_sample):
    # Selection failing because of out of bounds
    with pytest.raises(ValueError):
        interp(
            fixed_sample,
            b = Nearest(xr.DataArray([-2]), tolerance=1e-4),
        )


def test_oob_sel_nearest(fixed_sample):
    # This should pass
    interp(
        fixed_sample,
        b = Nearest(xr.DataArray([-2]), tolerance=None)
    )


def test_oob_interp(fixed_sample):
    # Interpolation failing because of out of bounds
    with pytest.raises(ValueError):
        interp(
            fixed_sample,
            b = Linear(xr.DataArray([-2]))
        )
        interp(
            fixed_sample,
            b = Spline(xr.DataArray([-2]))
        )
    with pytest.raises(ValueError):
        interp(
            fixed_sample,
            b = Linear(xr.DataArray([10])),
        )
        interp(
            fixed_sample,
            b = Spline(xr.DataArray([10])),
        )

    interp(
        fixed_sample,
        b = Linear(xr.DataArray([9])),
    )
    interp(
        fixed_sample,
        b = Spline(xr.DataArray([9])),
    )


def test_oob_interp_clip(fixed_sample):
    # Passes because clip=True
    interp(
        fixed_sample,
        b=Linear(xr.DataArray([-2]), bounds="clip"),
    )
    interp(
        fixed_sample,
        b=Spline(xr.DataArray([-2]), bounds="clip"),
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
    

@pytest.mark.parametrize('spacing', ["regular", "irregular"])
@pytest.mark.parametrize('kind', [Linear, Spline])
def test_interp_2D(request, spacing, kind):
    
    A = xr.DataArray(np.eye(3), dims=['x', 'y'])
    N = 100

    interpolated = interp(
        A,
        x=kind(
            xr.DataArray(np.linspace(-1, 3, N), dims=["new_x"]),
            bounds="clip",
            spacing=spacing,
        ),
        y=kind(
            xr.DataArray(np.linspace(-1, 3, N), dims=["new_Y"]),
            bounds="nan",
            spacing=spacing,
        ),
    )

    plt.imshow(interpolated)
    conftest.savefig(request)


@pytest.mark.parametrize("spacing", ["regular", "irregular"])
def test_spline(request, spacing):
    if spacing == "regular" :
        Y = xr.DataArray([1.0, 1.0, 1.0, 1.0], dims=["X"], coords=[np.array([0,1,2,3])])
    else :
        Y = xr.DataArray([1.0, 1.0, 1.0], dims=["X"], coords=[np.array([0,1,3])])
        
        assert np.isclose(interp(Y, X=Spline(xr.DataArray([0.5]), tension=0.5, spacing=spacing)), 1.0)
        assert np.isclose(interp(Y, X=Spline(xr.DataArray([1.5]), tension=0.5, spacing=spacing)), 1.0)
        
        Y = xr.DataArray([1.0, 1.0, 1.0, 1.0], dims=["X"], coords=[np.array([0,1,2,3])])
        assert np.isclose(interp(Y, X=Spline(xr.DataArray([0.5]), tension=0.5, spacing=spacing)), 1.0)
        assert np.isclose(interp(Y, X=Spline(xr.DataArray([1.5]), tension=0.5, spacing=spacing)), 1.0)
        assert np.isclose(interp(Y, X=Spline(xr.DataArray([2.5]), tension=0.5, spacing=spacing)), 1.0)
    

def test_inverse_func():
    indexer = Linear_Indexer([0., 1., 4., 9.], bounds="nan", spacing=np.sqrt)
    indexer(0.5)

    
@pytest.mark.parametrize('spacing', ["auto", lambda x: np.sqrt(x)])
def test_nearest_func(request, spacing):
    Y = xr.DataArray([1.0, 1.0, 1.0, 2.0], dims=["X"], coords=[np.array([0,1,4,16])])
    lp = np.linspace(0, 16, 1000)
    Ys = interp(Y, X=Nearest(xr.DataArray(lp), tolerance=None, spacing=spacing))
    plt.plot(Y.X, Y, 'ro')
    plt.plot(lp,Ys, 'b-')
    plt.grid()
    plt.show()
    conftest.savefig(request)
    plt.clf()
    
    
def test_nearest_func_values_invert(request):
    Y = xr.DataArray([1.0, 1.0, 1.0, 2.0], dims=["X"], coords=[np.array([0,1,4,16])])
    assert np.isclose(interp(Y, X=Nearest(xr.DataArray([8.9]), tolerance=None, spacing=lambda x: np.sqrt(x))), 1.0)
    assert np.isclose(interp(Y, X=Nearest(xr.DataArray([9.1]), tolerance=None, spacing=lambda x: np.sqrt(x))), 2.0)
    
    Y = xr.DataArray([1.0, 1.0, 1.0, 2.0], dims=["X"], coords=[np.array([0,1,4,16])])
    assert np.isclose(interp(Y, X=Nearest(xr.DataArray([9.9]), tolerance=None, spacing="auto")), 1.0)
    assert np.isclose(interp(Y, X=Nearest(xr.DataArray([10.1]), tolerance=None, spacing="auto")), 2.0)


@pytest.mark.parametrize("indexer_factory", **parametrize_dict({
    'nearest': lambda x: Nearest(x, tolerance=10),
    'linear_nan': lambda x: Linear(x, bounds="nan"),
    'spline_nan': lambda x: Spline(x, bounds="nan"),
    'linear_clip': lambda x: Linear(x, bounds="clip"),
    'linear_cycle': lambda x: Linear(x, bounds="cycle"),
    'spline_clip': lambda x: Spline(x, bounds="clip"),
}))
def test_interp_1D(request, indexer_factory):
    Y = xr.DataArray(
        [1.0, 2.0, 0.0, 3.0], dims=["X"], coords=[np.array([0.0, 1.0, 2.0, 3.0])]
    )
    Xi = xr.DataArray(np.linspace(-2, 6, 100))
    plt.plot(Y.X, Y, 'ro')
    Ys = interp(Y, X=indexer_factory(Xi))
    plt.plot(Xi.values, Ys.values, '-')
    plt.grid(True)
    conftest.savefig(request)
