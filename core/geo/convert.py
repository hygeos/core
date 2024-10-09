import numpy as np 


def convert_latlon(lat, lon):
    """
    Convert latitude and longitude vectors into 2D representation
    """
    
    lat_size = lat.shape
    lon_size = lon.shape
    assert len(lat_size) == 1 and len(lon_size) == 1, \
    "Latitude and longitude should be 1-dimensional"
    
    size = (lat_size[0], lon_size[0])
    new_lat = lat[:,np.newaxis]
    new_lat = np.repeat(new_lat, size[1], axis=1)
    new_lon = np.repeat(lon, size[0], axis=0)
    new_lon = new_lon.reshape(size).T
    
    return new_lat, new_lon