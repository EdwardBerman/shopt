# ---------------------------------------------------------#
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

# ---------------------------------------------------------#
fancyPrint("Handling Imports")
using Base: initarray!
using YAML
using BenchmarkTools
using Plots
using ForwardDiff
using LinearAlgebra
using PyCall
using Random
using Distributions
using SpecialFunctions
using Optim
using IterativeSolvers
using QuadGK
using DataFrames
using FFTW
using CSV
using Images, ImageFiltering
using Measures
using ProgressBars
using UnicodePlots
using Flux
using Flux.Optimise
using Flux: onehotbatch, throttle, @epochs, mse, msle
using CairoMakie

# ---------------------------------------------------------#
fancyPrint("Reading .jl Files")
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

# ---------------------------------------------------------#
#fancyPrint("Running Source Extractor")
# ---------------------------------------------------------#

fancyPrint("Processing Data for Fit")
starCatalog, errVignets, r, c, itr, u_coordinates, v_coordinates = cataloging(ARGS)
starCatalog = starCatalog[1:25]
#starCatalog, r,c, itr = catalogingWEBBPSF()
errVignets = errVignets[1:25]
u_coordinates = u_coordinates[1:25]
v_coordinates = v_coordinates[1:25]

itr = length(starCatalog)

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

    it = []
    loss = []

    function cb(opt_state:: Optim.OptimizationState)
      push!(it, opt_state.iteration)
      push!(loss, opt_state.value)
      return false  
    end
    global iteration = i
    try
      global x_cg = optimize(cost, 
                             g!, 
                             initial_guess, 
                             ConjugateGradient(),
                             Optim.Options(callback = cb))
      
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
      push!(failedStars, i)
      s_model[i] = 0
      g1_model[i] = 0
      g2_model[i] = 0
      continue
    end

    loss_time = Plots.plot(it, 
                           loss, 
                           linewidth = 5,
                           tickfontsize=8,
                           margin=15mm,
                           xlims=(0,30),
                           xguidefontsize=20,
                           yguidefontsize=20,
                           xlabel="Iteration", 
                           ylabel="Loss",
                           label="Star $i Model")
    push!(ltPlots, loss_time)

    if "$i" == "$itr"
      title = Plots.plot(title = "Analytic Profile Loss Vs Iteration (Model)",
                         titlefontsize=30,
                         grid = false, 
                         showaxis = false, 
                         bottom_margin = -50Plots.px)
        
      filler = Plots.plot(grid = false, 
                          showaxis = false, 
                          bottom_margin = -50Plots.px)
        
      Plots.savefig(Plots.plot(title,
                               filler,
                               ltPlots[1], 
                               ltPlots[2], 
                               ltPlots[3], 
                               ltPlots[4], 
                               ltPlots[5], 
                               ltPlots[6], 
                                  layout = (4,2),
                                  size = (1800,800)),
                                  joinpath("outdir", "lossTimeModel.pdf"))
    end
  end
end


println("\t \t Outliers in s: ", detect_outliers(s_model))
println("\n\t \t Outliers in g1: ", detect_outliers(g1_model))
println("\n\t \t Outliers in g2: ", detect_outliers(g2_model))

ns = length(detect_outliers(s_model))
ng1 = length(detect_outliers(g1_model))
ng2 = length(detect_outliers(g2_model))

println("\n\t \t Number of outliers in s: ", ns[1])
println("\n\t \t Number of outliers in g1: ", ng1[1])
println("\n\t \t Number of outliers in g2: ", ng2[1])

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
                    Dense(64, 128, leakyrelu),
                    Dense(128, r*c, tanh),
                   )
  
    # Define the full autoencoder
    autoencoder = Chain(encoder, decoder)

    # Define the loss function (mean squared error)
    #loss(nanToZero, autoencoder, x, x̂) = Flux.mse(nanToZero(autoencoder(x)), nanToZero(x̂))
    
    
    function relative_error_loss(x)
      relative_error = abs.(x - autoencoder(x)) ./ abs.(x .+ 1e-10)  # Add a small value to avoid division by zero
      mean(relative_error)
    end

    #x̂ = autoencoder(x)
    loss(x) = mse(autoencoder(x), x)
    #loss2(x) = mean(abs.(autoencoder(x) .- x)./abs.(x .+ 1e-10))
    #loss2(x) = mse(1,  sign(x)*sign(autoencoder(x))*abs.(autoencoder(x) ./ (x .+ 1e-10))) 
    #loss2(x; agg = mean, autoencoder=autoencoder) = agg( abs.((autoencoder(x) .- x) ./ (x .+ 1e-10)) * 100)
    #loss2(x) = msle(autoencoder(x), x)
    #loss2(x; agg = mean, eps = eps(eltype(autoencoder(x)))) =  agg((log.(abs.(autoencoder(x) .+ eps)) .- log.(abs.(x .+ eps))) .^ 2)

    optimizer = ADAM()

    
    # Format some random image data
    data = nanToZero(reshape(starCatalog[i], length(starCatalog[i])))
    
   
    # Train the autoencoder
    try
      for epoch in 1:epochs
        Flux.train!(loss, Flux.params(autoencoder), [(data,)], optimizer) #loss #Flux.params(autoencoder))
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

# ---------------------------------------------------------#
fancyPrint("Analytic Profile Fit for Learned Star")
#Copy Star Catalog then replace it with the learned pixel grid stars
@time begin
  pb = tqdm(1:length(starCatalog))
  for i in pb
    initial_guess = rand(3) #println("\t initial guess [σ, e1, e2]: ", initial_guess)
    set_description(pb, "Star $i/$(length(starCatalog)) Complete")
     
    it = []
    loss = []

    function cb(opt_state:: Optim.OptimizationState)
      push!(it, opt_state.iteration)
      push!(loss, opt_state.value)
      return false  
    end
    global iteration = i
    try 
      global y_cg = optimize(costD, 
                             gD!, 
                             initial_guess, 
                             ConjugateGradient(),
                             Optim.Options(callback = cb))
    
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
      push!(failedStars, i)
      s_data[i] = 0
      g1_data[i] = 0
      g2_data[i] = 0
      continue
    end

    loss_time = Plots.plot(it, 
                           loss, 
                           margin=15mm,
                           linewidth = 5,
                           tickfontsize=8,
                           xlims=(0,30),
                           xlabel="Iteration",
                           xguidefontsize=20,               
                           yguidefontsize=20,
                           ylabel="Loss",
                           label="Star $i Data")
    push!(ltdPlots, loss_time)
    
    if "$i" == "$itr"
      title = Plots.plot(title = "Analytic Profile Loss Vs Iteration (Data)", 
                         titlefontsize=30,
                         grid = false, 
                         showaxis = false, 
                         bottom_margin = -50Plots.px)

      filler = Plots.plot(grid = false, 
                          showaxis = false, 
                          bottom_margin = -50Plots.px)
      Plots.savefig(Plots.plot(title,
                               filler,
                               ltdPlots[1], 
                               ltdPlots[2], 
                               ltdPlots[3], 
                               ltdPlots[4], 
                               ltdPlots[5], 
                               ltdPlots[6], 
                               layout = (4,2),
                               size = (1800,800)),
                               joinpath("outdir", "lossTimeData.pdf"))
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
fancyPrint("Transforming (x,y) -> (u,v) | Interpolation Across the Field of View")

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

println("\n** Adding a Progress Bar Dramatically Increases the Run Time, but note that Interpolation across the FOV is taking place! **\n")

PolynomialMatrix = ones(r,c, 10)
@time begin
  for i in 1:r
    #pb = tqdm(1:c)
    for j in 1:c
      #set_description(pb, "Working on Pixel ($i, $j)")
      p_tuples = []
      for k in 1:length(pixelGridFits)
        push!(p_tuples, (u_coordinates[k], v_coordinates[k], pixelGridFits[k][i, j]))
      end

      function interpCostP(p; truth=p_tuples)
        I(u, v) = p[1]*u^3 + p[2]*v^3 + p[3]*u^2*v + p[4]*v^2*u + p[5]*u^2 + p[6]*v^2 + p[7]*u*v + p[8]*u + p[9]*v + p[10]
        t = truth
        function sumLoss(f, t)
          totalLoss = 0
          for i in 1:length(t)  #t = [(u,v, I), ...     ]
            totalLoss += (f(t[i][1], t[i][2]) - t[i][3])^2
          end
          return totalLoss
        end
        return sumLoss(I, t)
      end

      function polyG_P!(storage, p)
        grad_cost = ForwardDiff.gradient(interpCostP, p)
        storage[1] = grad_cost[1]
        storage[2] = grad_cost[2]
        storage[3] = grad_cost[3]
        storage[4] = grad_cost[4]
        storage[5] = grad_cost[5]
        storage[6] = grad_cost[6]
        storage[7] = grad_cost[7]
        storage[8] = grad_cost[8]
        storage[9] = grad_cost[9]
        storage[10] = grad_cost[10]
      end

      p_fov = optimize(interpCostP, polyG_P!, rand(10), ConjugateGradient())
      pC = p_fov.minimizer

      #Create Optimization Scheme with Truh Values from the PixelGridFits
      PolynomialMatrix[i,j,1] = pC[1]
      PolynomialMatrix[i,j,2] = pC[2]
      PolynomialMatrix[i,j,3] = pC[3]
      PolynomialMatrix[i,j,4] = pC[4]
      PolynomialMatrix[i,j,5] = pC[5]
      PolynomialMatrix[i,j,6] = pC[6]
      PolynomialMatrix[i,j,7] = pC[7]
      PolynomialMatrix[i,j,8] = pC[8]
      PolynomialMatrix[i,j,9] = pC[9]
      PolynomialMatrix[i,j,10] = pC[10]
    end
  end 

end
#println("Polynomial Matrix: $(PolynomialMatrix)")


# ---------------------------------------------------------#
fancyPrint("Plotting")

meanRelativeError = []
for i in 1:length(starCatalog)
  a = starCatalog[i]
  b = pixelGridFits[i]
  RelativeError = []
  for j in 1:size(starCatalog[i], 1)
    for k in 1:size(starCatalog[i], 2)
      push!(RelativeError, abs.(starCatalog[i][j,k] .- pixelGridFits[i][j,k]) ./ abs.(starCatalog[i][j,k] .+ 1e-10))
    end
  end
  push!(meanRelativeError, mean(RelativeError))
end

function sample_indices(array, k)
    indices = collect(1:length(array))  # Create an array of indices
    return sample(indices, k, replace = false)
end

sampled_indices = sort(sample_indices(starCatalog, 3))

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

Residuals = a - b
costSquaredError = Residuals.^2 

fft_image = fft(complex.(Residuals))
fft_image = abs2.(fft_image)

pk = []
for i in 1:10
  radius = range(1, max(r/2, c/2) - 1, length=10)
  push!(pk, powerSpectrum(fft_image, radius[i]))
end


#ksMatrix , b = ks 

fftmin = minimum(fft_image)
fftmax = maximum(fft_image)

cmx = maximum([maximum(a), maximum(b)])
cmn = minimum([minimum(a), minimum(b)])

plot_hm()
plot_hist()
plot_err()

a = nanMask2(a)
b = nanMask2(b)

#=
println(UnicodePlots.heatmap(get_middle_15x15(a), cmax = cmx, cmin = cmn, colormap=:inferno, title="Heatmap of star $starSample"))
println(UnicodePlots.heatmap(get_middle_15x15(b), cmax = cmx, cmin = cmn, colormap=:inferno, title="Heatmap of Pixel Grid Fit $starSample"))
println(UnicodePlots.heatmap(get_middle_15x15(a - b), colormap=:inferno, title="Heatmap of Residuals"))
=#

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
ax1 = fig1[1, 1] = CairoMakie.Axis(fig1, xlabel = L"u", ylabel = L"v")
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
                    labelsize = 14,
                    ticklabelsize = 14)
 
CairoMakie.colgap!(fig1.layout, 5)
 
save(joinpath("outdir", "s_uv.png"), fig1)

testField(u, v) = Point2f(dg1_du(u,v), dg1_dv(u,v)) # x'(t) = -x, y'(t) = 2y
u = range(minimum(u_coordinates), stop=maximum(u_coordinates), step=0.0001)            
v = range(minimum(v_coordinates), stop=maximum(v_coordinates), step=0.0001)            

g1_map = [g1(u,v) for u in u, v in v]
 
fig2 = Figure(resolution = (1920, 1080), fontsize = 30, fonts = (;regular="CMU Serif"))
ax2 = fig2[1, 1] = CairoMakie.Axis(fig2, xlabel = L"u", ylabel = L"v")
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
                    labelsize = 14,
                    ticklabelsize = 14)
 
CairoMakie.colgap!(fig2.layout, 5)
 
save(joinpath("outdir", "g1_uv.png"), fig2)

testField(u, v) = Point2f(dg2_du(u,v), dg2_dv(u,v)) # x'(t) = -x, y'(t) = 2y
u = range(minimum(u_coordinates), stop=maximum(u_coordinates), step=0.0001)            
v = range(minimum(v_coordinates), stop=maximum(v_coordinates), step=0.0001)            

g2_map = [g2(u,v) for u in u, v in v]

fig3 = Figure(resolution = (1920, 1080), fontsize = 30, fonts = (;regular="CMU Serif"))
ax3 = fig3[1, 1] = CairoMakie.Axis(fig3, xlabel = L"u", ylabel = L"v")
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
                    labelsize = 14,
                    ticklabelsize = 14)
 
CairoMakie.colgap!(fig3.layout, 5)
 
save(joinpath("outdir", "g2_uv.png"), fig3)

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



let
    # cf. https://github.com/JuliaPlots/Makie.jl/issues/822#issuecomment-769684652
    # with scale argument that is required now
    struct LogMinorTicks end
    
    function MakieLayout.get_minor_tickvalues(::LogMinorTicks, scale, tickvalues, vmin, vmax)
        vals = Float64[]
        for (lo, hi) in zip(
                  @view(tickvalues[1:end-1]),
                  @view(tickvalues[2:end]))
                        interval = hi-lo
                        steps = log10.(LinRange(10^lo, 10^hi, 11))
            append!(vals, steps[2:end-1])
        end
        vals
    end
    custom_formatter(values) = map(v -> "10" * Makie.UnicodeFun.to_superscript(round(Int64, v    )), values)
      data = a
      starData = b
      fig = Figure(resolution = (1800, 1800))
      ax_a, hm = CairoMakie.heatmap(fig[1, 1], log10.(data),
      axis=(; xminorticksvisible=true,
         xminorticks=IntervalsBetween(9)))
      ax_a.xlabel = "U"
      ax_a.ylabel = "V"
      ax_a.aspect = DataAspect()

      ax_b, hm = CairoMakie.heatmap(fig[1, 2], log10.(starData),
      axis=(; xminorticksvisible=true,
         xminorticks=IntervalsBetween(9)))
      ax_b.xlabel = "U"
      ax_b.ylabel = "V"
      ax_b.aspect = DataAspect()
    
      ax_c, hm = CairoMakie.heatmap(fig[1, 3], log10.(abs.(data - starData)),
      axis=(; xminorticksvisible=true,
         xminorticks=IntervalsBetween(9)))
      ax_c.xlabel = "U"
      ax_c.ylabel = "V"
      ax_c.aspect = DataAspect()
      
      cb = Colorbar(fig[1, 4], hm;
      tickformat=custom_formatter,
      minorticksvisible=true,
      minorticks=LogMinorTicks())
      ax_a.title = "Log Scale Model PSF"
      ax_b.title = "Log Scale Learned PSF"
      ax_c.title = "Log Scale Absolute Value of Residuals"
      save(joinpath("outdir", "logScale.pdf"), fig)
      save(joinpath("outdir", "logScale.png"), fig)
end

# ---------------------------------------------------------#
fancyPrint("Saving Data to summary.shopt")
writeData(s_model, g1_model, g2_model, s_data, g1_data, g2_data)
println(readData())

println(UnicodePlots.boxplot(["s model", "s data", "g1 model", "g1 data", "g2 model", "g2 data"], 
                             [s_model, s_data, g1_model, g1_data, g2_model, g2_data],
                            title="Boxplot of df.shopt"))
writeFitsData()

# ---------------------------------------------------------#
fancyPrint("Done! =]")

