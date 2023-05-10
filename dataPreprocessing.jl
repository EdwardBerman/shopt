using PyCall

function dataprocessing()
  py"""
  import webbpsf
  nc = webbpsf.NIRCam()
  psf = nc.calc_psf(nlambda=5, fov_arcsec=2)
  a = psf[1].data
  """
  julia_array = convert(Array{Float64,2}, py"a")
  rows, cols = size(julia_array)
  return julia_array, rows, cols
end
data, r, c = dataprocessing()


