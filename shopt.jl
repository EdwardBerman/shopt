# ---------------------------------------------------------#
@time begin
  include("argparser.jl")
  include("fancyPrint.jl")

  try
    process_arguments(ARGS)
  catch err
    println("Error: ", err)
    println("Usage: julia shopt.jl <configdir> <outdir> <catalog> <sci>")
    exit(1)
  end

  configdir = ARGS[1]
  outdir = ARGS[2]
  catalog = ARGS[3]
  sci = ARGS[4]

  if isdir(outdir)
    println("\tOutdir found")
  else
    println("\tOutdir not found, creating...")
    mkdir(outdir)
  end
end
# ---------------------------------------------------------#
fancyPrint("Handling Imports")
@time begin
  using Base: initarray!
  #println("\t\t    Base: initarray! imported")
  using YAML
  #println("\t\t    YAML imported")
  using BenchmarkTools
  #println("\t\t    BenchmarkTools imported")
  using Plots
  #println("\t\t    Plots imported")
  using ForwardDiff
  #println("\t\t    ForwardDiff imported")
  using LinearAlgebra
  #println("\t\t    LinearAlgebra imported")
  using PyCall
  #println("\t\t    PyCall imported")
  using Random
  #println("\t\t    Random imported")
  using Distributions
  #println("\t\t    Distributions imported")
  using SpecialFunctions
  #println("\t\t    SpecialFunctions imported")
  using Optim
  #println("\t\t    Optim imported")
  using IterativeSolvers
  #println("\t\t    IterativeSolvers imported")
  using QuadGK
  #println("\t\t    QuadGK imported")
  using DataFrames
  #println("\t\t    DataFrames imported")
  using FFTW
  #println("\t\t    FFTW imported")
  using CSV
  #println("\t\t    CSV imported")
  using Images, ImageFiltering
  #println("\t\t    Images, ImageFiltering imported")
  using Measures
  #println("\t\t    Measures imported")
  using ProgressBars
  #println("\t\t    ProgressBars imported")
  using UnicodePlots
  #println("\t\t    UnicodePlots imported")
  using Flux
  #println("\t\t    Flux imported")
  using Flux.Optimise
  #println("\t\t    Flux.Optimise imported")
  using Flux.Losses
  #println("\t\t    Flux.Losses imported")
  using Flux: onehotbatch, throttle, @epochs, mse, msle
  #println("\t\t    Flux: onehotbatch, throttle, @epochs, mse, msle imported")
  using CairoMakie
  #println("\t\t    CairoMakie imported")
  using Dates 
  #println("\t\t    Dates imported")
end
# ---------------------------------------------------------#
fancyPrint("Reading .jl Files")
@time begin
  include("plot.jl")
  include("analyticCGD.jl")
  include("radialProfiles.jl")
  include("pixelGridCGD.jl")
  include("masks.jl")
  include("dataPreprocessing.jl")
  include("outliers.jl")
  include("dataOutprocessing.jl")
  include("powerSpectrum.jl")
  include("kaisserSquires.jl")
  include("webbpsfProcessing.jl")
  include("interpolate.jl")
end
# ---------------------------------------------------------#
#fancyPrint("Running Source Extractor")
# ---------------------------------------------------------#

fancyPrint("Processing Data for Fit")
@time begin
  
  starCatalog, errVignets, r, c, itr, u_coordinates, v_coordinates = cataloging(ARGS)
  starCatalog = starCatalog
  errVignets = errVignets
  u_coordinates = u_coordinates
  v_coordinates = v_coordinates
  itr = length(starCatalog)
  
  
  #=
  starCatalog, r,c, itr = catalogingWEBBPSF()
  u_coordinates = rand(2)
  v_coordinates = rand(2)

  itr = length(starCatalog)
  =#

  #=
  starCatalog, r, c, u_coordinates, v_coordinates = gridPSFS() #return catalogNew, rows, cols, u_coords, v_coords
  errVignets = starCatalog
  itr = length(starCatalog)
  =#
end

starData = zeros(r, c)

A_model = zeros(itr)
s_model = zeros(itr)
g1_model = zeros(itr)
g2_model = zeros(itr)

A_data = zeros(itr)
s_data = zeros(itr)
g1_data = zeros(itr)
g2_data = zeros(itr)

ltPlots = []
failedStars = []
# ---------------------------------------------------------#
fancyPrint("Analytic Profile Fit for Model Star")
@time begin
  pb = tqdm(1:itr)
  for i in pb
    initial_guess = rand(3) #println("\t initial guess [σ, e1, e2]: ", initial_guess)
    set_description(pb, "Star $i/$itr Complete")
    
    #=
    it = []
    loss = []

    function cb(opt_state:: Optim.OptimizationState)
      push!(it, opt_state.iteration)
      push!(loss, opt_state.value)
      return false  
    end
    =#
    global iteration = i
    try
      global x_cg = optimize(cost, 
                             g!, 
                             initial_guess, 
                             LBFGS(),#ConjugateGradient()
                             Optim.Options(g_tol = 1e-6))#Optim.Options(callback = cb)
      
      s_model[i] = x_cg.minimizer[1]^2
      e1_guess = x_cg.minimizer[2]
      e2_guess = x_cg.minimizer[3]

      ellipticityData = sqrt((e1_guess)^2 + (e2_guess)^2)
      normGdata = sqrt(1 + 0.5*( (1/ellipticityData^2) - sqrt( (4/ellipticityData^2) + (1/ellipticityData^4)  )  )) 
      ratioData = ellipticityData/normGdata
      g1_model[i] = e1_guess/ratioData            
      g2_model[i] = e2_guess/ratioData  
  
      norm_data = zeros(r,c)
      for u in 1:r
        for v in 1:c
          norm_data[u,v] = fGaussian(u, v, g1_model[i], g2_model[i], s_model[i], r/2, c/2)
        end
      end
      A_model[i] = 1/sum(norm_data)
    catch
      println("Star $i failed")
      #push!(failedStars, i)
      s_model[i] = 0
      g1_model[i] = 0
      g2_model[i] = 0
      continue
    end
  end
end


#println("\t \t Outliers in s: ", detect_outliers(s_model))
#println("\n\t \t Outliers in g1: ", detect_outliers(g1_model))
#println("\n\t \t Outliers in g2: ", detect_outliers(g2_model))

ns = length(detect_outliers(s_model))
ng1 = length(detect_outliers(g1_model))
ng2 = length(detect_outliers(g2_model))

#println("\n\t \t Number of outliers in s: ", ns[1])
#println("\n\t \t Number of outliers in g1: ", ng1[1])
#println("\n\t \t Number of outliers in g2: ", ng2[1])

s_blacklist = []
for i in 1:length(s_model)
  if (s_model[i] < 0.075 || s_model[i] > 1) #i in failedStars is optional Since Failed Stars are assigned s=0 
    push!(s_blacklist, i)
  end
end

println("\nBlacklisted Stars: ", s_blacklist)
println("\nBlacklisted $(length(s_blacklist)) stars on the basis of s < 0.075 or s > 1 (Failed Stars Assigned 0)" )

for i in sort(s_blacklist, rev=true)
  splice!(starCatalog, i)
  splice!(errVignets, i)
  splice!(s_model, i)
  splice!(g1_model, i)
  splice!(g2_model, i)
  splice!(u_coordinates, i)
  splice!(v_coordinates, i)
end

failedStars = []

# ---------------------------------------------------------#
fancyPrint("Pixel Grid Fit")
pixelGridFits = []
@time begin
  pb = tqdm(1:length(starCatalog))
  for i in pb
    set_description(pb, "Star $i/$(length(starCatalog)) Complete")
    global iteration = i
    encoder = Chain(
                    Dense(r*c, 128, leakyrelu),
                    Dense(128, 64, leakyrelu),
                    Dense(64, 32, leakyrelu),
                   )
    # Define the decoder
    decoder = Chain(
                    Dense(32, 64, leakyrelu),
                    Dense(64, 128, leakyrelu), #leakyrelu #relu
                    Dense(128, r*c, tanh),   #tanh #sigmoid
                   )
    
    # Define the full autoencoder
    autoencoder = Chain(encoder, decoder)

    #x̂ = autoencoder(x)
    loss(x) = mse(autoencoder(x), x)
    
    function relative_error_loss(x)
      #x = nanToZero(nanMask(x))
      relative_error = abs.(x - autoencoder(x)) ./ abs.(x .+ 1e-10)  # Add a small value to avoid division by zero
      mean(relative_error)
    end

    # Define the optimizer
    optimizer = ADAM()

    
    # Format some random image data
    data = nanToZero(reshape(starCatalog[i], length(starCatalog[i])))
    
   
    # Train the autoencoder
    try
      min_gradient = 1e-5
      for epoch in 1:epochs
        Flux.train!(loss, Flux.params(autoencoder), [(data,)], optimizer) #loss#Flux.params(autoencoder))
        grad = Flux.gradient(Flux.params(autoencoder)) do
          loss(data)
        end
        grad_norm = norm(grad)
        if (grad_norm < min_gradient) #min_gradient
          #println(epoch)
          break
        end
      end
      # Take a sample input image
      input_image = reshape(starCatalog[i], length(starCatalog[i]))
  
      # Pass the input image through the autoencoder to get the reconstructed image
      reconstructed_image = autoencoder(input_image)

      #pg = optimize(pgCost, pg_g!, rand(r*c), ConjugateGradient())
      #push!(pixelGridFits ,reshape(pg.minimizer, (r, c)))
      pgf_current = reshape(reconstructed_image, (r, c))./sum(reshape(reconstructed_image, (r, c)))
      push!(pixelGridFits, pgf_current)
    catch ex
      println(ex)
      println("Star $i failed")
      push!(failedStars, i)
      push!(pixelGridFits, zeros(r,c))
      continue
    end

 
  
  end
end


ltdPlots = []
println("failed stars:", failedStars)
# ---------------------------------------------------------#
fancyPrint("Analytic Profile Fit for Learned Star")
#Copy Star Catalog then replace it with the learned pixel grid stars
@time begin
  pb = tqdm(1:length(starCatalog))
  for i in pb
    initial_guess = rand(3) #println("\t initial guess [σ, e1, e2]: ", initial_guess)
    set_description(pb, "Star $i/$(length(starCatalog)) Complete")
    
    #=
    it = []
    loss = []

    function cb(opt_state:: Optim.OptimizationState)
      push!(it, opt_state.iteration)
      push!(loss, opt_state.value)
      return false  
    end
    =#
    global iteration = i
    try 
      global y_cg = optimize(costD, 
                             gD!, 
                             initial_guess,
                             LBFGS(),#ConjugateGradient()ConjugateGradient(),
                             Optim.Options(g_tol = 1e-6)) #Optim.Options(callback = cb)
    
      s_data[i] = y_cg.minimizer[1]^2
      e1_guess = y_cg.minimizer[2]
      e2_guess = y_cg.minimizer[3]

      ellipticityData = sqrt((e1_guess)^2 + (e2_guess)^2)
      normGdata = sqrt(1 + 0.5*( (1/ellipticityData^2) - sqrt( (4/ellipticityData^2) + (1/ellipticityData^4)  )  )) 
      ratioData = ellipticityData/normGdata
      g1_data[i] = e1_guess/ratioData            
      g2_data[i] = e2_guess/ratioData  
      
      norm_data = zeros(r,c)
      for u in 1:r
        for v in 1:c
          norm_data[u,v] = fGaussian(u, v, g1_data[i], g2_data[i], s_data[i], r/2, c/2)
        end
      end
      A_data[i] = 1/sum(norm_data)

      if s_data[i] < 0.075 || s_data[i] > 1 
        push!(failedStars, i) 
      end
    catch
      println("Star $i failed")
      s_data[i] = 0
      g1_data[i] = 0
      g2_data[i] = 0
      continue
    end
    #println("\t Found A: ", A_data[i], "\t s: ", s_data[i]^2, "\t g1: ", g1_data[i], "\t g2: ", g2_data[i])

  end
end


println("failed stars: ", unique(failedStars))
println("\nRejected $(length(unique(failedStars))) more stars for failing or having either s < 0.075 or s > 1 when fitting an analytic profile to an autoencoded image. NB: These failed stars are being indexed after the blacklisted stars were removed.")

failedStars = unique(failedStars)

for i in sort(failedStars, rev=true)
  splice!(pixelGridFits, i)
  splice!(s_data, i)
  splice!(g1_data, i)
  splice!(g2_data, i)
  splice!(s_model, i)
  splice!(g1_model, i)
  splice!(g2_model, i)
  splice!(u_coordinates, i)
  splice!(v_coordinates, i)
  splice!(starCatalog, i)
  splice!(errVignets, i)
end

# ---------------------------------------------------------#
fancyPrint("Transforming (x,y) -> (u,v) | Interpolation [s, g1, g2] Across the Field of View")

s_data = s_data[1:length(pixelGridFits)]
g1_data = g1_data[1:length(pixelGridFits)]
g2_data = g2_data[1:length(pixelGridFits)]

s_tuples = []
for i in 1:length(starCatalog)
  push!(s_tuples, (u_coordinates[i], v_coordinates[i], s_data[i]))
end

s_fov = optimize(interpCostS, polyG_s!, rand(10), ConjugateGradient())
sC = s_fov.minimizer
println("\ns(u,v) = $(sC[1])u^3 + $(sC[2])v^3 + $(sC[3])u^2v + $(sC[4])v^2u + $(sC[5])u^2 + $(sC[6])v^2 + $(sC[7])uv + $(sC[8])u + $(sC[9])v + $(sC[10])\n")

s(u,v) = sC[1]*u^3 + sC[2]*v^3 + sC[3]*u^2*v + sC[4]*v^2*u + sC[5]*u^2 + sC[6]*v^2 + sC[7]*u*v + sC[8]*u + sC[9]*v + sC[10]
ds_du(u,v) = sC[1]*3*u^2 + sC[3]*2*u*v + sC[4]*v^2 + sC[5]*2*u + sC[7]*v + sC[8]
ds_dv(u,v) = sC[2]*3*v^2 + sC[3]*u^2 + sC[4]*2*u*v + sC[6]*2*v + sC[7]*u + sC[9]

g1_tuples = []
for i in 1:length(starCatalog)
  push!(g1_tuples, (u_coordinates[i], v_coordinates[i], g1_data[i]))
end

g1_fov = optimize(interpCostg1, polyG_g1!, rand(10), ConjugateGradient())
g1C = g1_fov.minimizer
println("\ng1(u,v) = $(g1C[1])u^3 + $(g1C[2])v^3 + $(g1C[3])u^2v + $(g1C[4])v^2u + $(g1C[5])u^2 + $(g1C[6])v^2 + $(g1C[7])uv + $(g1C[8])u + $(g1C[9])v + $(g1C[10])\n")

g1(u,v) = g1C[1]*u^3 + g1C[2]*v^3 + g1C[3]*u^2*v + g1C[4]*v^2*u + g1C[5]*u^2 + g1C[6]*v^2 + g1C[7]*u*v + g1C[8]*u + g1C[9]*v + g1C[10]
dg1_du(u,v) = g1C[1]*3*u^2 + g1C[3]*2*u*v + g1C[4]*v^2 + g1C[5]*2*u + g1C[7]*v + g1C[8]
dg1_dv(u,v) = g1C[2]*3*v^2 + g1C[3]*u^2 + g1C[4]*2*u*v + g1C[6]*2*v + g1C[7]*u + g1C[9]

g2_tuples = []
for i in 1:length(starCatalog)
  push!(g2_tuples, (u_coordinates[i], v_coordinates[i], g2_data[i]))
end
h_uv_data = g2_tuples

g2_fov = optimize(interpCostg2, polyG_g2!, rand(10), ConjugateGradient())
g2C = g2_fov.minimizer
println("\ng2(u,v) = $(g2C[1])u^3 + $(g2C[2])v^3 + $(g2C[3])u^2v + $(g2C[4])v^2u + $(g2C[5])u^2 + $(g2C[6])v^2 + $(g2C[7])uv + $(g2C[8])u + $(g2C[9])v + $(g2C[10])\n")

g2(u,v) = g2C[1]*u^3 + g2C[2]*v^3 + g2C[3]*u^2*v + g2C[4]*v^2*u + g2C[5]*u^2 + g2C[6]*v^2 + g2C[7]*u*v + g2C[8]*u + g2C[9]*v + g2C[10]
dg2_du(u,v) = g2C[1]*3*u^2 + g2C[3]*2*u*v + g2C[4]*v^2 + g2C[5]*2*u + g2C[7]*v + g2C[8]
dg2_dv(u,v) = g2C[2]*3*v^2 + g2C[3]*u^2 + g2C[4]*2*u*v + g2C[6]*2*v + g2C[7]*u + g2C[9]

#println("\n** Adding a Progress Bar Dramatically Increases the Run Time, but note that Interpolation across the FOV is taking place! **\n")

PolynomialMatrix = ones(r,c, 10)
  
function sample_indices(array, k)
  indices = collect(1:length(array))  # Create an array of indices
  return randperm(length(indices))[1:k] #sample(indices, k, replace = false)
end

total_samples = length(pixelGridFits)
training_ratio = 0.8
training_samples = round(Int, training_ratio * total_samples)

training_indices = sample_indices(pixelGridFits, training_samples)
training_stars = pixelGridFits[training_indices]
training_u_coords = u_coordinates[training_indices]
training_v_coords = v_coordinates[training_indices]

validation_indices = setdiff(1:total_samples, training_indices)
validation_stars = pixelGridFits[validation_indices]
validation_u_coords = u_coordinates[validation_indices]
validation_v_coords = v_coordinates[validation_indices]
validation_star_catalog = starCatalog[validation_indices]

fancyPrint("Transforming (x,y) -> (u,v) | Interpolation Pixel by Pixel Across the Field of View")

@time begin
  for i in 1:r
    #pb = tqdm(1:c)
    for j in 1:c #1:c #pb
      #set_description(pb, "Working on Pixel ($i , $j)")
      
      function objective_function(p, x, y, degree)
        num_coefficients = (degree + 1) * (degree + 2) ÷ 2
        value = 0
        counter = 0
        for i in 1:(degree + 1)
          for j in 1:(degree + 1)
            if (i - 1) + (j - 1) <= degree
              counter += 1
              value += p[counter] * x^(i - 1) * y^(j - 1) #Make p 2-D then flatten
            end
          end
        end

        return value
      end

      function polynomial_optimizer(degree, x_data, y_data, z_data)
        num_coefficients = (degree + 1) * (degree + 2) ÷ 2
        initial_guess = ones(num_coefficients)  # Initial guess for the coefficients
        function objective(p)
          epsilon = 1e-8  # Small constant to avoid division by zero or negative infinity
          loss = sum((objective_function(p, x_val, y_val, degree) - z_actual)^2  for ((x_val, y_val), z_actual) in zip(zip(x_data, y_data), z_data) if !isnan(z_actual))
          #loss = sum(log(1 + (objective_function(p, x_val, y_val, degree) - z_actual)^2) for ((x_val, y_val), z_actual) in zip(zip(x_data, y_data), z_data) if !isnan(z_actual))
          #loss = sum((objective_function(p, x_val, y_val, degree) - z_actual)^2 for ((x_val, y_val), z_actual) in zip(zip(x_data, y_data), z_data) if !isnan(z_actual))
          return loss
        end
 
        result = optimize(objective, initial_guess, autodiff=:forward, LBFGS(), Optim.Options(f_tol=1e-40)) #autodiff=:forward

        return Optim.minimizer(result)
      end

      #degree = 3
      x_data = training_u_coords  # Sample x data
      y_data = training_v_coords  # Sample y data
      z_data = []  # Sample z data
      for k in 1:length(training_stars)
        push!(z_data, training_stars[k][i, j])
      end

      pC = polynomial_optimizer(degree, x_data, y_data, z_data)
      #println("Optimized Polynomial Coefficients:")
      #println(pC)

      #Create Optimization Scheme with Truh Values from the PixelGridFits
      for k in 1:length(pC)
        PolynomialMatrix[i,j,k] = pC[k]
      end
    end
  end 

end
#=
println("Optimized Polynomial Coefficients:")
println(coefficients)


=#

#println("Polynomial Matrix: $(PolynomialMatrix)")

#println(PolynomialMatrix[31,31,:])
# ---------------------------------------------------------#
fancyPrint("Plotting")
@time begin
  meanRelativeError = []
  for i in 1:length(starCatalog)
    RelativeError = []
    for j in 1:size(starCatalog[i], 1)
      for k in 1:size(starCatalog[i], 2)
        push!(RelativeError, abs.(starCatalog[i][j,k] .- pixelGridFits[i][j,k]) ./ abs.(starCatalog[i][j,k] .+ 1e-10))
      end
    end
    push!(meanRelativeError, mean(RelativeError))
  end


  sampled_indices = sort(sample_indices(starCatalog, 3))
  #sampled_indices = [1,2]

  println("Sampled indices: ", sampled_indices)

  function get_middle_15x15(array::Array{T, 2}) where T
      rows, cols = size(array)
      row_start = div(rows, 2) - 7
      col_start = div(cols, 2) - 7
      
      return array[row_start:(row_start+14), col_start:(col_start+14)]
  end

  starSample = rand(1:length(starCatalog))
  a = starCatalog[starSample]
  b = pixelGridFits[starSample]

  a = nanToZero(a)
  b = nanToZero(b)

  cmx = maximum([maximum(a), maximum(b)])
  cmn = minimum([minimum(a), minimum(b)])

  println(UnicodePlots.heatmap(get_middle_15x15(a), cmax = cmx, cmin = cmn, colormap=:inferno, title="Heatmap of star $starSample"))
  println(UnicodePlots.heatmap(get_middle_15x15(b), cmax = cmx, cmin = cmn, colormap=:inferno, title="Heatmap of Pixel Grid Fit $starSample"))
  println(UnicodePlots.heatmap(get_middle_15x15(a - b), colormap=:inferno, title="Heatmap of Residuals"))


  #=
  println(UnicodePlots.histogram(s_model, vertical=true, title="Histogram of s model"))
  println(UnicodePlots.histogram(s_data, vertical=true, title="Histogram of s data"))
  println(UnicodePlots.histogram(g1_model, vertical=true, title="Histogram of g1 model"))
  println(UnicodePlots.histogram(g1_data, vertical=true, title="Histogram of g1 data"))
  println(UnicodePlots.histogram(g2_model, vertical=true, title="Histogram of g2 model"))
  println(UnicodePlots.histogram(g2_data, vertical=true, title="Histogram of g2 data"))
  =#

  testField(u, v) = Point2f(ds_du(u,v), ds_dv(u,v)) # x'(t) = -x, y'(t) = 2y
  u = range(minimum(u_coordinates), stop=maximum(u_coordinates), step=0.0001)            
  v = range(minimum(v_coordinates), stop=maximum(v_coordinates), step=0.0001)            

  s_map = [s(u,v) for u in u, v in v]
   
  fig1 = Figure(resolution = (1920, 1080), fontsize = 30, fonts = (;regular="CMU Serif"))
  ax1 = fig1[1, 1] = CairoMakie.Axis(fig1, xlabel = L"u", ylabel = L"v",xticklabelsize = 40, yticklabelsize = 40, xlabelsize = 40, ylabelsize = 40)
  fs1 = CairoMakie.heatmap!(ax1, u, v, s_map, colormap = Reverse(:plasma))
  CairoMakie.streamplot!(ax1,
              testField,
              u,
              v,
              colormap = Reverse(:plasma),
              gridsize = (32, 32),
              density = 0.25,
              arrow_size = 10)

  CairoMakie.Colorbar(fig1[1, 2],
                      fs1,
                      label = L"s(u,v)",
                      width = 20,
                      labelsize = 50,
                      ticklabelsize = 14)
   
  CairoMakie.colgap!(fig1.layout, 5)
   
  save(joinpath("outdir", "s_uv.png"), fig1)

  testField(u, v) = Point2f(dg1_du(u,v), dg1_dv(u,v)) # x'(t) = -x, y'(t) = 2y
  u = range(minimum(u_coordinates), stop=maximum(u_coordinates), step=0.0001)            
  v = range(minimum(v_coordinates), stop=maximum(v_coordinates), step=0.0001)            

  g1_map = [g1(u,v) for u in u, v in v]
   
  fig2 = Figure(resolution = (1920, 1080), fontsize = 30, fonts = (;regular="CMU Serif"))
  ax2 = fig2[1, 1] = CairoMakie.Axis(fig2, xlabel = L"u", ylabel = L"v",xticklabelsize = 40, yticklabelsize = 40, xlabelsize = 40, ylabelsize = 40)
  fs2 = CairoMakie.heatmap!(ax2, u, v, g1_map, colormap = Reverse(:plasma))
  CairoMakie.streamplot!(ax2,
              testField,
              u,
              v,
              colormap = Reverse(:plasma),
              gridsize = (32, 32),
              density = 0.25,
              arrow_size = 10)

  CairoMakie.Colorbar(fig2[1, 2],
                      fs2,
                      label = L"g1(u,v)",
                      width = 20,
                      labelsize = 50,
                      ticklabelsize = 14)
   
  CairoMakie.colgap!(fig2.layout, 5)
   
  save(joinpath("outdir", "g1_uv.png"), fig2)

  testField(u, v) = Point2f(dg2_du(u,v), dg2_dv(u,v)) # x'(t) = -x, y'(t) = 2y
  u = range(minimum(u_coordinates), stop=maximum(u_coordinates), step=0.0001)            
  v = range(minimum(v_coordinates), stop=maximum(v_coordinates), step=0.0001)            

  g2_map = [g2(u,v) for u in u, v in v]

  fig3 = Figure(resolution = (1920, 1080), fontsize = 30, fonts = (;regular="CMU Serif"))
  ax3 = fig3[1, 1] = CairoMakie.Axis(fig3, xlabel = L"u", ylabel = L"v", xticklabelsize = 40, yticklabelsize = 40, xlabelsize = 40, ylabelsize = 40)
  fs3 = CairoMakie.heatmap!(ax3, u, v, g2_map, colormap = Reverse(:plasma))
  CairoMakie.streamplot!(ax3,
              testField,
              u,
              v,
              colormap = Reverse(:plasma),
              gridsize = (32, 32),
              density = 0.25,
              arrow_size = 10)

  CairoMakie.Colorbar(fig3[1, 2],
                      fs3,
                      label = L"g2(u,v)",
                      width = 20,
                      labelsize = 50,
                      ticklabelsize = 14)
   
  CairoMakie.colgap!(fig3.layout, 5)
   
  save(joinpath("outdir", "g2_uv.png"), fig3)

  #=
  scale = 1/0.29
  ks93, k0 = ks(g1_map, g2_map)
  ksCosmos = get_middle_15x15(imfilter(ks93, Kernel.gaussian(scale)))
  kshm = Plots.heatmap(ksCosmos,
                       title="Kaisser-Squires", 
                       xlabel="u", 
                       ylabel="v",
                       xlims=(0.5, size(ksCosmos, 2) + 0.5),  # set the x-axis limits to include the full cells
                       ylims=(0.5, size(ksCosmos, 1) + 0.5),  # set the y-axis limits to include the full cells
                       aspect_ratio=:equal,
                       ticks=:none,  # remove the ticks
                       frame=:box,  # draw a box around the plot
                       grid=:none,  # remove the grid lines
                       size=(1920,1080))

  Plots.savefig(kshm, joinpath("outdir","kaisserSquires.png"))
  =#
end

# ---------------------------------------------------------#
fancyPrint("Saving Data to summary.shopt")
#writeData(s_model, g1_model, g2_model, s_data, g1_data, g2_data)
#println(readData())

println(UnicodePlots.boxplot(["s model", "s data", "g1 model", "g1 data", "g2 model", "g2 data"], 
                             [s_model, s_data, g1_model, g1_data, g2_model, g2_data],
                            title="Boxplot of df.shopt"))

#errVignets = []
#for i in 1:2
 # push!(errVignets, rand(161,161))
#end
writeFitsData()

# ---------------------------------------------------------#
fancyPrint("Done! =]")

