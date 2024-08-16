import pytest

import numpy as np

from core.geo.convert import convert_latlon


def test_latlon():
    lat = np.linspace(-10,20,300)
    lon = np.linspace(-30,30,600)

    lat, lon = convert_latlon(lat, lon)
    
    assert lat.shape == lon.shape