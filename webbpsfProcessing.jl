using PyCall
using Statistics
using Distributions

function catalogingWEBBPSF()
  py"""
  import numpy as np
  from astropy.io import fits
  from astropy.table import Table, hstack, vstack
  import webbpsf
  insts = ['NIRCam','NIRCam','NIRCam','NIRCam']
  filts = ['F115W', 'F150W', 'F277W', 'F444W']

  psfs = {}
  for i, (instname, filt) in enumerate(zip(insts, filts)):
    inst = webbpsf.instrument(instname)
    inst.filter = filt
    psf = inst.calc_psf(fov_arcsec=5.0)
    psfs[instname+filt] = psf
  
  psf1 = psfs['NIRCamF115W'][1].data
  psf2 = psfs['NIRCamF150W'][1].data
  psf3 = psfs['NIRCamF277W'][1].data
  psf4 = psfs['NIRCamF444W'][1].data

  """
  julia_array1 = convert(Array{Float64,2}, py"psf1")
  julia_array2 = convert(Array{Float64,2}, py"psf2")
  rows, cols = size(julia_array1)
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

  push!(catalogNew, blur(julia_array1))
  push!(catalogNew, blur(julia_array2))
  return catalogNew, rows, cols, 2
end

function gridPSFS()
  py"""
  import numpy as np
  from astropy.io import fits
  from astropy.table import Table, hstack, vstack
  import webbpsf
  nir = webbpsf.NIRCam()
  nir.filter = "F115W"
  nir.detector = "NRCA2"
  f115w = nir.psf_grid(all_detectors=False)
  
  f115wdata = []
  u = []
  v = []

  #for j in range(len(f115w)):
  for i in range(len(f115w.data)): #len(f115w[j].data)):
    f115wdata.append(f115w.data[i]) #f115wdata.append(f115w[j].data[i])
    u.append(f115w.meta['grid_xypos'][i][0]) #u.append(f115w[j].meta['grid_xypos'][i][0])
    v.append(f115w.meta['grid_xypos'][i][1]) #v.append(f115w[j].meta['grid_xypos'][i][1])

  f115wdata = np.array(f115wdata)

  #nir.filter = "F150W"
  #f150w = np.array(nir.psf_grid()[0])
  #nir.filter = "F277W"
  #f277w = np.array(nir.psf_grid()[0])
  #nir.filter = "F444W"
  #f444w = np.array(nir.psf_grid()[0])
  """
  
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
  
  catalogNew = []
  f115w = convert(Array{Float64,3}, py"f115wdata")
  u_coords = convert(Array{Float64,1}, py"u")
  v_coords = convert(Array{Float64,1}, py"v")
  for i in 1:size(f115w, 1)
    push!(catalogNew, blur(f115w[i,:,:]))
  end
  #len = convert(Int64, py"F115W") + convert(Int64, py"F150W") + convert(Int64, py"F277W") + convert(Int64, py"F444W")
  #for i in 1:len
    #filtered_array = [filter(!isnan, row) for row in eachrow(catalog[i])]
   # push!(catalogNew, blur(julia_array))
  #end
  rows, cols = size(catalogNew[1])
  return catalogNew, rows, cols, u_coords, v_coords
end

