#=
Load in Config shopt.yml
=#
nt = nthreads()
fancyPrintTwo("Julia is using $nt threads to run ShOpt.jl. To change this, on UNIX run export JULIA_NUM_THREADS=[Integer] and on Windows set JULIA_NUM_THREADS=[Integer].") 
fancyPrintTwo("For example: export JULIA_NUM_THREADS=4 or set JULIA_NUM_THREADS=4")
config = YAML.load_file(joinpath(configdir, "shopt.yml"))
epochs = config["NNparams"]["epochs"]
degree = config["polynomialDegree"]
new_img_dim = config["stampSize"]
r, c = new_img_dim, new_img_dim
snrCutoff = config["dataProcessing"]["SnRPercentile"]
YAMLSAVE = config["saveYaml"]
minAnalyticGradientModel = config["AnalyticFitParams"]["minGradientAnalyticModel"]
minAnalyticGradientLearned = config["AnalyticFitParams"]["minGradientAnalyticLearned"]
AnalyticStampSize = config["AnalyticFitParams"]["analyticFitStampSize"]
minPixelGradient = config["NNparams"]["minGradientPixel"]
UnicodePlotsPrint = config["plots"]["unicodePlots"]
pythonPlots = config["plots"]["pythonPlots"]
parametersHistogram = config["plots"]["normalPlots"]["parametersHistogram"]
parametersScatterplot = config["plots"]["normalPlots"]["parametersScatterplot"]
cairomakiePlots = config["plots"]["cairomakiePlots"]["streamplots"]
sLowerBound = config["dataProcessing"]["sLowerBound"]
sUpperBound = config["dataProcessing"]["sUpperBound"]
comments = config["CommentsOnRun"]
unity_sum = config["sum_pixel_grid_and_inputs_to_unity"]
training_ratio = config["training_ratio"]
summary_name = config["summary_name"]
mode = config["mode"] # Options: auotoencoder, lanczos
PCAterms = config["PCAterms"]
polynomial_interpolation_stopping_gradient = config["polynomial_interpolation_stopping_gradient"]
lanczos = config["lanczos"]
truncate_summary_file = config["truncate_summary_file"]
iterationsPolyfit = config["iterations"]
alpha = config["alpha"]
chisq_stopping_gradient = config["chisq_stopping_gradient"]

#=
Log these config choices
=#

println("Key Config Choices:")
println("━ Mode: ", mode)
println("━ α:", alpha)
println("━ Chisq Stopping Gradient:", chisq_stopping_gradient)
println("━ Iterations: ", iterationsPolyfit)
println("━ Summary Name: ", summary_name)
println("━ PCA Terms: ", PCAterms)
println("━ Lanczos n: ", lanczos)
println("━ Polynomial Interpolation Stopping Gradient: ", polynomial_interpolation_stopping_gradient)
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
println("━ Python Plots: ", pythonPlots)
println("━ Parameters Histogram: ", parametersHistogram)
println("━ Parameters Scatterplot: ", parametersScatterplot)
println("━ Stream Plots: ", cairomakiePlots)
println("━ s Lower Bound: ", sLowerBound)
println("━ s Upper Bound: ", sUpperBound)
println("━ Training Ratio: ", training_ratio)
println("━ Sum Pixel Grid and Inputs to Unity: ", unity_sum)
println("━ Truncate Summary File: ", truncate_summary_file)
println("━ Comments: ", comments)

#=
Utility Function used to count NaNs in an image, was used in testing and may be useful for future debugging in the data preprocessing process
=#
function countNaNs(arr)
    count = 0
    
    for element in arr
        if isnan(element)
            count += 1
        end
    end
    
    return count
end

function sinc(x)
    if x == 0
        return 1
    else
        return sin(x) / x
    end
end

function lanczos_kernel(x, a)
    if abs(x) >= a
        return 0
    elseif x == 0
        return 1
    else
        return a * sinc(pi*x) * sinc(pi*x/a)
    end
end

function smooth(data, a)
    kernel = [lanczos_kernel(i, a) * lanczos_kernel(j, a) for i in -a:a, j in -a:a]
    kernel = kernel ./ sum(kernel)  # Normalize the kernel

    nx, ny = size(data)
    smoothed_data = copy(data)  # Initialize with original data to keep NaN positions

    for i in 1:nx
        for j in 1:ny
            if isnan(data[i, j])
                continue  # Skip smoothing for NaN values
            end
            norm_factor = 0.0  # This is to normalize considering non-NaN values
            smoothed_value = 0.0
            for di in max(1, i-a):min(nx, i+a)
                for dj in max(1, j-a):min(ny, j+a)
                    if !isnan(data[di, dj])
                        smoothed_value += kernel[di-i+a+1, dj-j+a+1] * data[di, dj]
                        norm_factor += kernel[di-i+a+1, dj-j+a+1]
                    end
                end
            end
            smoothed_data[i, j] = smoothed_value / norm_factor
        end
    end

    return smoothed_data
end

function oversample_image(image, new_dim)
  oversampled_image = zeros(new_dim, new_dim)
  scale_factor = 1/(new_dim / size(image)[1])
  for y in 1:new_dim
    for x in 1:new_dim
      src_x = (x - 0.5) * scale_factor + 0.5
      src_y = (y - 0.5) * scale_factor + 0.5
      x0 = max(1, floor(Int, src_x))
      x1 = min(size(image)[1], x0 + 1)
      y0 = max(1, floor(Int, src_y))
      y1 = min(size(image)[2], y0 + 1)
      dx = src_x - x0
      dy = src_y - y0
      oversampled_image[y, x] =
        (1 - dx) * (1 - dy) * image[y0, x0] +
        dx * (1 - dy) * image[y0, x1] +
        (1 - dx) * dy * image[y1, x0] +
        dx * dy * image[y1, x1]
    end
  end
  return oversampled_image
end
  
function undersample_image(image, new_dim)
  undersampled_image = imresize(image, new_dim, new_dim, algorithm =:bicubic)
  return undersampled_image
end

function sample_image(image, new_dim)
  if new_dim > size(image)[1]
    return oversample_image(image, new_dim)
  else
    return undersample_image(image, new_dim)
  end
end
#=
function undersample_image(image, new_dim)
    # Get the original dimensions of the image
    height, width = size(image)

    # Create Lanczos interpolation object
    interp = interpolate(image, Lanczos(4))

    # Calculate the scaling factors
    scale_height = (height - 1) / (new_dim - 1)
    scale_width = (width - 1) / (new_dim - 1)

    # Generate coordinates for the new image
    x_coords = 1:new_dim
    y_coords = 1:new_dim

    # Perform downsampling with Lanczos interpolation
    undersampled_image = zeros(new_dim, new_dim)

    for y in 1:new_dim
        for x in 1:new_dim
            undersampled_image[y, x] = interp(scale_height * (y - 1) + 1, scale_width * (x - 1) + 1)
        end
    end

    return undersampled_image
end
=#
#=
Supply an array and the new dimension you want and get the middle nxn of that array
=#
function get_middle_nxn(array, n)
    rows, cols = size(array)

    # Calculate the starting row and column indices
    row_start = div(rows - n, 2) + 1
    col_start = div(cols - n, 2) + 1

    # Return the sub-matrix
    return array[row_start:(row_start + n - 1), col_start:(col_start + n - 1)]
end

#=
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
=#

#=
SnR function used to calculate the signal to noise ratio of an image
=#

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

function cataloging(args; nm=nanMask, nz=nanToZero, snr=signal_to_noise, dout=outliers_filter, alpha=alpha)
  catalog = args[3]

  py"""
  import numpy as np
  from astropy.io import fits
  
  python_datadir = $catalog
  mode = $mode
  print(python_datadir)
  alpha_default = $alpha
  print(alpha_default)
  print("Making error cutouts...")
  f = fits.open(python_datadir)
  def find_extension_with_colname(fits_file, target_colname):
    with fits.open(fits_file) as hdulist:
        matching_extensions = []
        
        for idx, hdu in enumerate(hdulist):
            if isinstance(hdu, fits.BinTableHDU) or isinstance(hdu, fits.TableHDU):
                if target_colname in hdu.columns.names:
                    matching_extensions.append(idx)
                    
        return matching_extensions
  idx = find_extension_with_colname(python_datadir, 'VIGNET')[0]
  
  vignets = f[idx].data['VIGNET']
  
  if mode == "chisq":
    backgrounds = f[idx].data['BACKGROUND']
    sigma_bs = [np.sqrt(np.var(background)) for background in backgrounds]
  

  def sigma_squared_i(pi, g, sigma_b, alpha):
    return sigma_b + pi / g + (alpha * pi)**2

  def compute_sigma_i_for_vignet(vignet, sigma_b, g=4, alpha=alpha_default, eta=1.0):
    sigma_i_cutout = np.zeros_like(vignet, dtype=float)
    for i in range(vignet.shape[0]):
      for j in range(vignet.shape[1]):
        pi = vignet[i, j]
        sigma_i_cutout[i, j] = sigma_squared_i(pi, g, sigma_b, alpha)
    return sigma_i_cutout

  # Compute sigma_i for each vignette using its respective sigma_b
  if mode == "chisq":
    err_vignets = [compute_sigma_i_for_vignet(vignet, sigma_b) for vignet, sigma_b in zip(vignets, sigma_bs)]
  #err_vignets = f[idx].data['ERR_VIGNET']
  l = len(vignets)

  u = f[idx].data['ALPHAWIN_J2000'] 
  v = f[idx].data['DELTAWIN_J2000']

  snr = f[2].data['SNR_WIN']
  """

  datadir = py"python_datadir"
  v = py"vignets"
  
  if mode == "chisq"
    err = py"err_vignets"
  end
  
  catalog = py"list(map(np.array, $v))"
  
  if mode == "chisq"
    errVignets = py"list(map(np.array, $err))"
  end
  
  u_coords = convert(Array{Float64,1}, py"u")
  v_coords = convert(Array{Float64,1}, py"v")
  snr = convert(Array{Float32,1}, py"snr")

  r = size(catalog[1], 1)
  c = size(catalog[1], 2)
  
  catalogNew = []
  signal2noiseRatios = []
  for i in 1:length(catalog)
    if unity_sum
      push!(catalogNew, nm(catalog[i])./sum(nz(nm(catalog[i]))))
    else
      push!(catalogNew, nm(catalog[i]))
    end
    #push!(catalogNew, nm(catalog[i])./sum(nz(nm(catalog[i]))))
    #push!(signal2noiseRatios, snr(catalog[i], errVignets[i]))
  end
  
  new_snr_arr = Array{Float64}(undef, length(snr))
  for (i, element) in enumerate(snr)
    new_snr_arr[i] = element
  end
  new_snr_arr = new_snr_arr[.!isnan.(new_snr_arr)] 
  
  println(UnicodePlots.boxplot(["snr"], [new_snr_arr], title="signal to noise ratio"))


  catalogNew, outlier_indices = dout(snr, catalogNew, snrCutoff)
  
  if mode == "chisq"
    errVignets = [errVignets[i] for i in 1:length(errVignets) if i ∉ outlier_indices]
  end

  println("━ Number of vignets: ", length(catalog))
  println("━ Removed $(length(catalog) - length(catalogNew)) outliers on the basis of Signal to Noise Ratio")
 #= 
  for i in 1:length(catalogNew)
    catalogNew[i] = get_middle_nxn(catalogNew[i], new_img_dim)
    errVignets[i] = get_middle_nxn(errVignets[i], new_img_dim)
  end
  println("━ Sampled all vignets to $(new_img_dim) x $(new_img_dim) from $(r) x $(c) via cropping")
  r = size(catalogNew[1], 1)
  c = size(catalogNew[1], 2)
  =#

  if new_img_dim/size(catalogNew[1], 1) !=1
    if new_img_dim/size(catalogNew[1], 1) < 1
      for i in 1:length(catalogNew)
        catalogNew[i] = smooth(nm(get_middle_nxn(catalogNew[i], new_img_dim))/sum(nz(nm(get_middle_nxn(catalogNew[i], new_img_dim)))), lanczos)
        if mode == "chisq"
          errVignets[i] = get_middle_nxn(errVignets[i], new_img_dim)
        end
      end
    else
      for i in 1:length(catalogNew)
        catalogNew[i] = smooth(nm(oversample_image(catalogNew[i], new_img_dim))/sum(nz(nm(oversample_image(catalogNew[i], new_img_dim)))), lanczos)
        if mode == "chisq"
          errVignets[i] = oversample_image(errVignets[i], new_img_dim)
        end
      end
    end
  end
  
  println("━ Sampled all vignets to $(size(catalogNew[1], 1)) x $(size(catalogNew[1], 2)) from $(r) x $(c) via over/under sampling")
  r = size(catalogNew[1], 1)
  c = size(catalogNew[1], 2)
  k = rand(1:length(catalogNew))
  if UnicodePlotsPrint
    println(UnicodePlots.heatmap(nz(nm(get_middle_nxn(catalogNew[k],15))), colormap=:inferno, title="Sampled Vignet $k")) 
  end
  
  if mode == "chisq"
    return catalogNew, r, c, length(catalogNew), u_coords, v_coords, outlier_indices, errVignets
  else 
    return catalogNew, r, c, length(catalogNew), u_coords, v_coords, outlier_indices
  end
end


