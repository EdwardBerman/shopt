config = YAML.load_file(joinpath(configdir, "shopt.yml"))
epochs = config["NNparams"]["epochs"]
degree = config["polynomialDegree"]
new_img_dim = config["stampSize"]
snrCutoff = config["dataProcessing"]["SnRPercentile"]
YAMLSAVE = config["saveYaml"]
minAnalyticGradientModel = config["AnalyticFitParams"]["minGradientAnalyticModel"]
minAnalyticGradientLearned = config["AnalyticFitParams"]["minGradientAnalyticLearned"]
AnalyticStampSize = config["AnalyticFitParams"]["analyticFitStampSize"]
minPixelGradient = config["NNparams"]["minGradientPixel"]
UnicodePlotsPrint = config["plots"]["unicodePlots"]
parametersHistogram = config["plots"]["normalPlots"]["parametersHistogram"]
parametersScatterplot = config["plots"]["normalPlots"]["parametersScatterplot"]
cairomakiePlots = config["plots"]["cairomakiePlots"]["streamplots"]
sLowerBound = config["dataProcessing"]["sLowerBound"]
sUpperBound = config["dataProcessing"]["sUpperBound"]
comments = config["CommentsOnRun"]
training_ratio = config["training_ratio"]

println("Key Config Choices:")
println("━ Max Epochs: ", epochs)
println("━ Polynomial Degree: ", degree)
println("━ Stamp Size: ", new_img_dim)
println("━ Signal to Noise Ratio Cutoff: ", snrCutoff)
println("━ Save YAML: ", YAMLSAVE)
println("━ Stopping Analytic Fit Gradient Star Vignets: ", minAnalyticGradientModel)
println("━ Stopping Analytic Fit Gradient Learned Vignets: ", minAnalyticGradientLearned)
println("━ Analytic Fit Stamp Size: ", AnalyticStampSize)
println("━ Stopping Pixel Fit Gradient: ", minPixelGradient)
println("━ Print Unicode Plots: ", UnicodePlotsPrint)
println("━ Parameters Histogram: ", parametersHistogram)
println("━ Parameters Scatterplot: ", parametersScatterplot)
println("━ Stream Plots: ", cairomakiePlots)
println("━ s Lower Bound: ", sLowerBound)
println("━ s Upper Bound: ", sUpperBound)
println("━ Training Ratio: ", training_ratio)
println("━ Comments: ", comments)

function countNaNs(arr)
    count = 0
    
    for element in arr
        if isnan(element)
            count += 1
        end
    end
    
    return count
end

function get_middle_nxn(array, n)
  rows, cols = size(array)
  if isodd(rows)
    if isodd(n)
      row_start = div(rows,2) - (n ÷ 2)
      col_start = div(cols,2) - (n ÷ 2)
      return array[row_start:(row_start + (2*(n ÷ 2))), col_start:(col_start + (2*(n ÷ 2)))]
    else
      array = array[1:(rows - 1), 1:(cols - 1)]
      rows, cols = size(array)
      row_start = div(rows,2) - (n ÷ 2)
      col_start = div(cols,2) - (n ÷ 2)
      return array[(1 + row_start):(row_start + (2*(n ÷ 2))), ( 1 + col_start):(col_start + (2*(n ÷ 2)))]
    end
  else
    if isodd(n)
      row_start = div(rows,2) - (n ÷ 2)
      col_start = div(cols,2) - (n ÷ 2)
      return array[row_start:(row_start + (2*(n ÷ 2))), col_start:(col_start + (2*(n ÷ 2)))]
    else
      row_start = div(rows,2) - (n ÷ 2)
      col_start = div(cols,2) - (n ÷ 2)
      return array[(1 + row_start):(row_start + (2*(n ÷ 2))), (1 + col_start):(col_start + (2*(n ÷ 2)))]
    end
  end
end


function signal_to_noise(I, e; nm=nanMask, nz=nanToZero)
  signal_power = sum(nz(nm(I)).^2)
  noise_power = sum(e.^2)
  snr = 10*log10(signal_power/noise_power)
  return snr
end

#=
Consider 
In [10]: len(f[2].data['DELTAWIN_J2000'])
Out[10]: 290
=#

function cataloging(args; nm=nanMask, nz=nanToZero, snr=signal_to_noise, dout=outliers_filter)
  catalog = args[3]
  sci_image = args[4]

  py"""
  import numpy as np
  from astropy.io import fits
  from astropy.table import Table, hstack, vstack
  from astropy.wcs import WCS
  from astropy import units as u
  from astropy.coordinates import SkyCoord
  
  python_datadir = $catalog
  f = fits.open(python_datadir)
  vignets = f[2].data['VIGNET']
  err_vignets = f[2].data['ERR_VIGNET']
  l = len(vignets)
  python_sci_file = $sci_image
  g = fits.open(python_sci_file)
  def find_wcs_info(filename):
    with fits.open(filename) as hdul:
      for idx, hdu in enumerate(hdul):
        try:
          wcs = WCS(hdu.header)
          return wcs, idx  # Return the WCS and extension index
        except Exception:
          continue  # Skip if WCS info is not present or invalid
        return None, None  # If no WCS info is found
   
  filename = python_sci_file
  wcs, extension_idx = find_wcs_info(filename)
  if wcs is not None:
    print(f"WCS info found in extension {extension_idx}:")
    print(wcs)
    # Additional WCS operations can be performed here
  else:
    print("No WCS info found in any extension.")
  
  w = wcs
  #w = WCS(g[0].header)

  x_coords = f[2].data['XWIN_IMAGE']
  y_coords = f[2].data['YWIN_IMAGE']
  pixel_coords = []
  for x, y in zip(x_coords, y_coords):
      pixel_coords.append([x, y])
  
  origin = 0
  ra_dec_coords = w.all_pix2world(pixel_coords, origin)
  u = ra_dec_coords[:,0]
  v = ra_dec_coords[:,1]
  """

  datadir = py"python_datadir"
  v = py"vignets"
  err = py"err_vignets"
  catalog = py"list(map(np.array, $v))"
  errVignets = py"list(map(np.array, $err))"
  uv_coords = convert(Array{Float64,2}, py"ra_dec_coords")
  u_coords = convert(Array{Float64,1}, py"u")
  v_coords = convert(Array{Float64,1}, py"v")

  r = size(catalog[1], 1)
  c = size(catalog[1], 2)
  
  catalogNew = []
  signal2noiseRatios = []
  for i in 1:length(catalog)
    push!(catalogNew, nm(catalog[i])./sum(nz(nm(catalog[i]))))
    push!(signal2noiseRatios, snr(catalog[i], errVignets[i]))
  end

  new_snr_arr = Array{Float64}(undef, length(signal2noiseRatios))
  for (i, element) in enumerate(signal2noiseRatios)
    new_snr_arr[i] = element
  end

  println(UnicodePlots.boxplot(["snr"], [new_snr_arr], title="signal to noise ratio"))


  catalogNew, errVignets = dout(signal2noiseRatios, catalogNew, errVignets, snrCutoff)
  println("━ Number of vignets: ", length(catalog))
  println("━ Removed $(length(catalog) - length(catalogNew)) outliers on the basis of Signal to Noise Ratio")
  
  for i in 1:length(catalogNew)
    catalogNew[i] = get_middle_nxn(catalogNew[i], new_img_dim)
    errVignets[i] = get_middle_nxn(errVignets[i], new_img_dim)
  end
  println("━ Sampled all vignets to $(new_img_dim) x $(new_img_dim) from $(r) x $(c)")
  r = size(catalogNew[1], 1)
  c = size(catalogNew[1], 2)
  
  return catalogNew, errVignets, r, c, length(catalogNew), u_coords, v_coords
end


