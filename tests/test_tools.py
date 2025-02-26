import pytest
import numpy as np
import xarray as xr
from core.tools import reglob, xr_filter_decorator, xr_unfilter, xr_filter


@pytest.mark.parametrize("regexp", [".*.py", "[a-z]+.*"])
def test_reglob(regexp):
    p = "core"
    l = reglob(p, regexp)
    assert len(l) != 0


def sample_dataset():
    return xr.Dataset(
        {
            "A": (("x", "y"), np.random.rand(4, 5)),
            "B": (("bands", "x", "y"), np.ones((3, 4, 5), dtype="int")),
        },
        coords={"bands": [400, 500, 600]},
    )


def test_xr_filter_unfilter():
    ds = sample_dataset()

    ok = ds.A < 0.5

    # Extract valid pixels
    sub = xr_filter(ds, ok)

    # Reformat sub to full array
    full = xr_unfilter(sub, ok)
    print(full)


@pytest.mark.parametrize("transparent", [True, False])
def test_xr_filter_decorator(transparent: bool):
    def myfunc(ds: xr.Dataset, a=None) -> xr.Dataset:
        return xr.merge(
            [
                ds.A.rename("A1"),
                ds.B.rename("B1"),
            ]
        )

    ds = sample_dataset()

    res = xr_filter_decorator(
        0,
        lambda x: x.A < 0.5,
        fill_value_int=-1,
        transparent=transparent,
    )(myfunc)(ds)
    print(res)

    # Check that the coords are preserved
    assert len(ds.coords) == len(res.coords)
