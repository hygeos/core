import pytest
import numpy as np
import xarray as xr
from core.tools import reglob, xr_filter_decorator, xr_unfilter, xr_filter


@pytest.mark.parametrize("regexp", [".*.py", "[a-z]+.*"])
def test_reglob(regexp):
    p = "core"
    l = reglob(p, regexp)
    assert len(l) != 0


def sample_dataset(with_xy_coords: bool):
    coords = {"bands": [400, 500, 600]}
    if with_xy_coords:
        coords["x"] = [0, 1, 2, 3]
        coords["y"] = [0, 1, 2, 3, 4]

    return xr.Dataset(
        {
            "A": (("x", "y"), np.random.rand(4, 5)),
            "B": (("bands", "x", "y"), np.ones((3, 4, 5), dtype="int")),
        },
        coords=coords,
    )


@pytest.mark.parametrize("with_xy_coords", [True, False])
def test_xr_filter_unfilter(with_xy_coords: bool):
    ds = sample_dataset(with_xy_coords)

    ok = ds.A < 0.5

    # Extract valid pixels
    sub = xr_filter(ds, ok)

    # Reformat sub to full array
    full = xr_unfilter(sub, ok)
    
    # Check that the coords are preserved
    assert len(ds.coords) == len(full.coords)


@pytest.mark.parametrize("transparent", [True, False])
@pytest.mark.parametrize("with_xy_coords", [True, False])
def test_xr_filter_decorator(transparent: bool, with_xy_coords: bool):
    def myfunc(ds: xr.Dataset) -> xr.Dataset:
        return xr.merge(
            [
                ds.A.rename("A1"),
                ds.B.rename("B1"),
            ]
        )

    ds = sample_dataset(with_xy_coords)

    res = xr_filter_decorator(
        0,
        lambda x: x.A < 0.5,
        fill_value_int=-1,
        transparent=transparent,
    )(myfunc)(ds)
    print(res)

    # Check that the coords are preserved
    assert len(ds.coords) == len(res.coords)
