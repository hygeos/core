#!/usr/bin/env python3

from core.tools import Var


class names:
    
    # Dimensions
    rows        = Var("y", desc="Vertical dimension of rasters", units="pixels")
    columns     = Var("x", desc="Horizontal dimension of rasters", units="pixels")
    lon         = Var("longitude", desc="Horizontal earth coordinate axis", units="Degrees", minv=-180, maxv=180, dtype='float32')
    lat         = Var("latitude", desc="Vertical earth coordinate axis", units="Degrees", minv=-90, maxv=90, dtype='float32')

    bands       = Var("bands", desc="spectral dimension of the acquisition")
    bands_nvis  = Var("bands_nvis", desc="Visible spectral dimension of the acquisition")
    bands_ir    = Var("bands_ir", desc="Infrared spectral dimension of the acquisition")
    bnames      = Var("bandnames", desc="Name of bands")
    detector    = Var("detectors", desc="Index of detector")
    
    # Radiometry 
    rtoa        = Var("Rtoa", desc="Top of Atmosphere reflectance", dtype='float32')
    ltoa        = Var("Ltoa", desc="Top of Atmosphere radiance", units="W.m-2.sr-1")            
    bt          = Var("BT", desc="Brightness Temperature", units="K", minv=0)
    rho_w       = Var("rho_w", desc="Water Reflectance")

    wav         = Var("wav", desc="Effective wavelength", units="nm") 
    cwav        = Var("cwav", desc="Central (nominal) wavelength", units="nm")

    F0          = Var("F0", desc="Solar flux irradiance", units="W.m-2.sr-1")
    
    # Angles 
    sza = Var("sza", desc="Sun zenith angle",       units="Degrees", minv=   0, maxv= 90, dtype='float32')
    vza = Var("vza", desc="View zenith angle",      units="Degrees", minv=   0, maxv= 90, dtype='float32')
    saa = Var("saa", desc="Sun azimuth angle",      units="Degrees", minv=   0, maxv=180, dtype='float32')
    vaa = Var("vaa", desc="View azimuth angle",     units="Degrees", minv=   0, maxv=180, dtype='float32')
    raa = Var("raa", desc="Relative azimuth angle", units="Degrees", minv=-180, maxv=180, dtype='float32')
    mus = Var("mus", desc="Cosine of the sun zenith angle",  dtype='float32')
    muv = Var("muv", desc="Cosine of the view zenith angle", dtype='float32')
    
    # Flags
    flags   = Var("flags", desc="Bitmask describing the pixel data", dtype='uint16')
    quality = Var("quality", desc="Boolean mask describing the pixel quality", dtype='bool')
    
    # Attributes
    crs             = Var("crs", desc="Projection")
    unit            = Var("unit")
    datetime        = Var("datetime")
    platform        = Var("platform")
    sensor          = Var("sensor")
    shortname       = Var("shortname")
    resolution      = Var("resolution", desc="Resolution in meter")
    description     = Var("description")
    product_name    = Var("product_name")
    input_directory = Var("input_directory")

