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
using Flux: onehotbatch, throttle, @epochs, mse
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
fancyPrint("Running Source Extractor")
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
ns = length(detect_outliers(s_model))
ng1 = length(detect_outliers(g1_model))
ng2 = length(detect_outliers(g2_model))

println("\t \t Number of outliers in s: ", ns[1])
println("\t \t Number of outliers in g1: ", ng1[1])
println("\t \t Number of outliers in g2: ", ng2[1])

# ---------------------------------------------------------#
fancyPrint("Pixel Grid Fit")
pixelGridFits = []
@time begin
  pb = tqdm(1:itr)
  for i in pb
    set_description(pb, "Star $i/$itr Complete")
    global iteration = i
    encoder = Chain(
                    Dense(r*c, 128, relu),
                    Dense(128, 64, relu),
                    Dense(64, 32, relu),
                   )
    # Define the decoder
    decoder = Chain(
                    Dense(32, 64, relu),
                    Dense(64, 128, relu),
                    Dense(128, r*c, sigmoid),
                   )
  
    # Define the full autoencoder
    autoencoder = Chain(encoder, decoder)

    # Define the loss function (mean squared error)
    #loss(nanToZero, autoencoder, x, x̂) = Flux.mse(nanToZero(autoencoder(x)), nanToZero(x̂))
    
    
    function relative_error_loss(x)
      relative_error = abs.(x - autoencoder(x)) ./ (x .+ 1e-10)  # Add a small value to avoid division by zero
      mean(relative_error)
    end

    #x̂ = autoencoder(x)
    loss(x) = mse(autoencoder(x), x)
    #loss(x; model=autoencoder, filter = nanToZero) = mse(filter(model(x)), filter(x))

    optimizer = ADAM()

    
    # Format some random image data
    data = reshape(starCatalog[i], length(starCatalog[i]))
    
   
    # Train the autoencoder
    try
      for epoch in 1:1000
        Flux.train!(loss, Flux.params(autoencoder), [(data,)], optimizer) #Flux.params(autoencoder))
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

println("failed stars: ", failedStars)
# ---------------------------------------------------------#
fancyPrint("Plotting")


failedStars = unique(failedStars)

for i in sort(failedStars, rev=true)
  splice!(starCatalog, i)
  splice!(errVignets, i)
  splice!(pixelGridFits, i)
  splice!(s_model, i)
  splice!(s_data, i)
  splice!(g1_model, i)
  splice!(g1_data, i)
  splice!(g2_model, i)
  splice!(g2_data, i)
  splice!(u_coordinates, i)
  splice!(v_coordinates, i)
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

starSample = rand(1:(itr - length(failedStars)))
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

# ---------------------------------------------------------#
fancyPrint("Transforming (x,y) -> (u,v) | Interpolation Across the Field of View")
s_tuples = []
for i in 1:length(s_data)
  push!(s_tuples, (u_coordinates[i], v_coordinates[i], s_data[i]))
end

h_uv_data = s_tuples

s_fov = optimize(interpCost, polyG!, rand(10), ConjugateGradient())
IC = s_fov.minimizer
println("IC: ", IC)

s(u,v) = IC[1]*u^3 + IC[2]*v^3 + IC[3]*u^2*v + IC[4]*v^2*u + IC[5]*u^2 + IC[6]*v^2 + IC[7]*u*v + IC[8]*u + IC[9]*v + IC[10]
ds_du(u,v) = IC[1]*3*u^2 + IC[3]*2*u*v + IC[4]*v^2 + IC[5]*2*u + IC[7]*v + IC[8]
ds_dv(u,v) = IC[2]*3*v^2 + IC[3]*u^2 + IC[4]*2*u*v + IC[6]*2*v + IC[7]*u + IC[9]

testField(u, v) = Point2f(ds_du(u,v), ds_dv(u,v)) # x'(t) = -x, y'(t) = 2y
u = range(minimum(u_coordinates), stop=maximum(u_coordinates), step=0.0001)            
v = range(minimum(v_coordinates), stop=maximum(v_coordinates), step=0.0001)            

z = [s(u,v) for u in u, v in v]
 
fig = Figure(resolution = (1920, 1080), fontsize = 30, fonts = (;regular="CMU Serif"))
ax = fig[1, 1] = CairoMakie.Axis(fig, xlabel = L"u", ylabel = L"v")
fs = CairoMakie.heatmap!(ax, u, v, z, colormap = Reverse(:plasma))
CairoMakie.streamplot!(ax,
            testField,
            u,
            v,
            colormap = Reverse(:plasma),
            gridsize = (32, 32),
            density = 0.25,
            arrow_size = 10)

CairoMakie.Colorbar(fig[1, 2],
                    fs,
                    label = L"s(u,v)",
                    width = 20,
                    labelsize = 14,
                    ticklabelsize = 14)
 
CairoMakie.colgap!(fig.layout, 5)
 
save(joinpath("outdir", "vectorfield.png"), fig)

g1_tuples = []
for i in 1:length(g1_data)
  push!(g1_tuples, (u_coordinates[i], v_coordinates[i], g1_data[i]))
end
h_uv_data = g1_tuples

# ---------------------------------------------------------#
fancyPrint("Saving DataFrame to df.shopt")
writeData(s_model, g1_model, g2_model, s_data, g1_data, g2_data)
println(readData())

println(UnicodePlots.boxplot(["s model", "s data", "g1 model", "g1 data", "g2 model", "g2 data"], 
                             [s_model, s_data, g1_model, g1_data, g2_model, g2_data],
                            title="Boxplot of df.shopt"))

# ---------------------------------------------------------#
fancyPrint("Done! =]")

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

