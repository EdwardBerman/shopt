using PyCall
py"""
import webbpsf
nc = webbpsf.NIRCam()
psf = nc.calc_psf(nlambda=5, fov_arcsec=2)
a = psf[1].data
"""

julia_array = convert(Array{Float64,2}, py"a")
print(size(julia_array))
