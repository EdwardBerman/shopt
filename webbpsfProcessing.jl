using PyCall

function catalogingWEBBPSF()
  py"""
  import numpy as np
  from astropy.io import fits
  from astropy.table import Table, hstack, vstack
  import webbpsf
  nc = webbpsf.NIRCam()
  psf = nc.calc_psf(nlambda=5, fov_arcsec=2)
  a = psf[1].data
  """
  julia_array = convert(Array{Float64,2}, py"a")
  rows, cols = size(julia_array)
  datadir = py"python_datadir"
  catalogNew = []
  
  function blur(img::Array{T, 2}) where T<:Real
    dummyArray = zeros(size(img, 1), size(img, 2))
    for u in 1:size(img, 1)
      for v in 1:size(img, 2)
        mu = 0.0
        sigma = 1
        normal_dist = Normal(mu, sigma)
        truncated_dist = Truncated(normal_dist, -0.1*img[u,v], 0.1*img[u,v])
        dummyArray[u,v] = img[u,v] + rand(truncated_dist)
      end
    end
    dummyArray = dummyArray./sum(dummyArray)
    return dummyArray
  end

  for i in 1:25 
    #filtered_array = [filter(!isnan, row) for row in eachrow(catalog[i])]
    push!(catalogNew, blur(julia_array))
  end
  return catalogNew, rows, cols, 25
end


