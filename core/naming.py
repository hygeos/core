# standard library imports
# ...
        
# third party imports
import xarray as xr
        
# sub package imports
from core.static import interface


class _name(object):
    
    @interface
    def __init__(self, 
                 name: str, 
                 desc: str = None, 
                 unit: str = None, 
                 minv: int|float = None, 
                 maxv: int|float = None,
                 dtype: str = None):
        self.name = name
        self.desc = desc
        self.unit = unit
        self.minv = minv
        self.maxv = maxv
        self.range = (minv, maxv)
        self.dtype = dtype
    
    def __str__(self):
        return self.name
    
class names:
    rows      = _name("y", "Vertical dimension of rasters", "pixels")
    columns   = _name("x", "Horizontal dimension of rasters", "pixels")
    lon       = _name("longitude", "Horizontal earth coordinate axis", "Degrees", -180, 180, 'float32')
    lat       = _name("latitude", "Vertical earth coordinate axis", "Degrees", -90, 90, 'float32')

    bands    = _name("bands", "Visible spectral dimension of the acquisition", None)
    bands_ir = _name("bands_ir", "Infrared spectral dimension of the acquisition", None)
    
    # Radiometry 
    rtoa     = _name("Rtoa", "Top of Atmosphere reflectance", None, dtype='float32')
    ltoa     = _name("Ltoa", "Top of Atmosphere radiance", "W.m-2.sr-1")               
    ltoa_ir  = _name("Ltoa_ir", "Top of Atmosphere radiance in infrared", "W.m-2.sr-1")
    
    bt       = _name("BT", "Brightness Temperature", "K", minv=0)
    rho_w    = _name("rho_w", "Water Reflectance", None)
    wav      = _name("wav", "Effective wavelength", "nm") 

    wav_ir   = _name("wav_ir", "Effective wavelength in infrared", "nm")
    cwav     = _name("cwav", "Central (nominal) wavelength", "nm")

    F0       = _name("F0", "Solar flux irradiance", "W.m-2.sr-1")
    
    # TODO CHECK min and max values ! (call François)
    sza = _name("sza", "Sun zenith angle",       "Degrees", minv=   0, maxv= 90, dtype='float32')
    vza = _name("vza", "View zenith angle",      "Degrees", minv=   0, maxv= 90, dtype='float32')
    saa = _name("saa", "Sun azimuth angle",      "Degrees", minv=   0, maxv=180, dtype='float32')
    vaa = _name("vaa", "View azimuth angle",     "Degrees", minv=   0, maxv=180, dtype='float32')
    raa = _name("raa", "Relative azimuth angle", "Degrees", minv=-180, maxv=180, dtype='float32')
    
    # Flags
    flags = _name("flags", "Bitmask describing the pixel data", None, dtype='uint16')

    # Attributes
    crs             = _name("crs", "Projection")
    datetime        = _name("datetime")
    platform        = _name("platform")
    sensor          = _name("sensor")
    shortname       = _name("shortname")
    resolution      = _name("resolution")
    description     = _name("description")
    product_name    = _name("product_name")
    input_directory = _name("input_directory")