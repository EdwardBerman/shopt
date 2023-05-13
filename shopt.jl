println("Handling Imports")
using Base: initarray!
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
using CSV

println("Importing Files")
include("plot.jl")
include("analyticCGD.jl")
include("radialProfiles.jl")
include("pixelGridCGD.jl")
include("dataPreprocessing.jl")
include("outliers.jl")
include("dataOutprocessing.jl")

println("Processing Data for Fit")
star, r, c = dataprocessing()

for u in 1:r
  for v in 1:c
    star[u,v] = star[u,v] + rand(Normal(-0.1*star[u,v], star[u,v]*0.1))
  end
end

 
starData = zeros(r, c)

itr = 6
A_model = zeros(itr)
s_model = zeros(itr)
g1_model = zeros(itr)
g2_model = zeros(itr)


A_data = zeros(itr)
s_data = zeros(itr)
g1_data = zeros(itr)
g2_data = zeros(itr)

ltPlots = []

println("Analytic Profile Fit for Model Star")
@time begin
  for i in 1:itr
    println("\t Star $i")
    initial_guess = rand(3)
    println("\t initial guess [σ, e1, e2]: ", initial_guess)
     
    it = []
    loss = []

    function cb(opt_state:: Optim.OptimizationState)
      push!(it, opt_state.iteration)
      push!(loss, opt_state.value)
      return false  
    end
    x_cg = optimize(cost, 
                    g!, 
                    initial_guess, 
                    ConjugateGradient(),
                    Optim.Options(callback = cb))

    loss_time = plot(it, 
                     loss, 
                     xlabel="Iteration", 
                     ylabel="Loss",
                     label="Star $i Data")
    push!(ltPlots, loss_time)
    
    if "$i" == "$itr"
      title = plot(title = "Analytic Profile Loss Vs Iteration (Model)", 
                   grid = false, 
                   showaxis = false, 
                   bottom_margin = -50Plots.px)

      filler = plot(grid = false, 
                    showaxis = false, 
                    bottom_margin = -50Plots.px)

      savefig(plot(title,
                   filler,
                   ltPlots[1], 
                   ltPlots[2], 
                   ltPlots[3], 
                   ltPlots[4], 
                   ltPlots[5], 
                   ltPlots[6], 
                   layout = (4,2),
                   size = (900,400)), 
                        joinpath("outdir", "lossTimeModel.png"))
    end

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
    println("\t Found A: ", A_model[i], "\t s: ", s_model[i]^2, "\t g1: ", g1_model[i], "\t g2: ", g2_model[i])

  end
end


println("\t \t Outliers in s: ", detect_outliers(s_model))
ns = length(detect_outliers(s_model))
ng1 = length(detect_outliers(g1_model))
ng2 = length(detect_outliers(g2_model))

println("\t \t Number of outliers in s: ", ns[1])
println("\t \t Number of outliers in g1: ", ng1[1])
println("\t \t Number of outliers in g2: ", ng2[1])

println("Pixel Grid Fit")
pg = optimize(pgCost, pg_g!, zeros(r*c), ConjugateGradient())
print(pg)
pg = reshape(pg.minimizer, (r, c))


ltdPlots = []

println("Analytic Profile Fit for Learned Star")
@time begin
  for i in 1:itr
    println("\t Star $i")
    initial_guess = rand(3)
    println("\t initial guess [σ, e1, e2]: ", initial_guess)
     
    it = []
    loss = []

    function cb(opt_state:: Optim.OptimizationState)
      push!(it, opt_state.iteration)
      push!(loss, opt_state.value)
      return false  
    end
    x_cg = optimize(costD, 
                    gD!, 
                    initial_guess, 
                    ConjugateGradient(),
                    Optim.Options(callback = cb))

    loss_time = plot(it, 
                     loss, 
                     xlabel="Iteration", 
                     ylabel="Loss",
                     label="Star $i Model")
    push!(ltdPlots, loss_time)
    
    if "$i" == "$itr"
    title = plot(title = "Analytic Profile Loss Vs Iteration (Data)", grid = false, showaxis = false, bottom_margin = -50Plots.px)
      filler = plot(grid = false, showaxis = false, bottom_margin = -50Plots.px)
      savefig(plot(title,
                   filler,
                   ltdPlots[1], 
                   ltdPlots[2], 
                   ltdPlots[3], 
                   ltdPlots[4], 
                   ltdPlots[5], 
                   ltdPlots[6], 
                   layout = (4,2),
                   size = (900,400)), 
                        joinpath("outdir", "lossTimeData.png"))
    end

    s_data[i] = x_cg.minimizer[1]^2
    e1_guess = x_cg.minimizer[2]
    e2_guess = x_cg.minimizer[3]

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
    println("\t Found A: ", A_data[i], "\t s: ", s_data[i]^2, "\t g1: ", g1_data[i], "\t g2: ", g2_data[i])

  end
end


# Plotting Heatmaps
print("\nPlotting \n")


norm2 = zeros(r, c)
norm2[5,5] = 1
norm2[5,6] = 1
norm2[6,5] = 1
norm2[6,6] = 1

for u in 1:r
  for v in 1:c
    norm2[u,v] = fGaussian(u, v, mean(g1_model), mean(g2_model), mean(s_model), r/2, c/2)
  end
end
A_model = 1/sum(norm2)

for u in 1:r
  for v in 1:c
    starData[u,v] = A_model*norm2[u,v]
  end
end

Residuals = star - starData
costSquaredError = Residuals.^2 
chiSquare = zeros(r, c)
for u in 1:r
  for v in 1:c
    chiSquare[u,v] = costSquaredError[u,v]/var(vec(star)) 
  end
end

amin = minimum([minimum(star), minimum(starData)])
amax = maximum([maximum(star), maximum(starData)])
csmin = minimum(chiSquare)
csmax = maximum(chiSquare)
rmin = minimum(Residuals)
rmax = maximum(Residuals)
csemin = minimum(costSquaredError)
csemax = maximum(costSquaredError)
rpgmin = minimum((star - pg).^2)
rpgmax = maximum((star - pg).^2)

dof = r*c - 3
p = 1 - cdf(Chisq(dof), sum(chiSquare))#ccdf = 1 - cdf
p = string(p)
println("p-value: ", p, "\n")


plot_hm(p)
plot_hist()
plot_err()

println("Saving DataFrame to df.shopt")
writeData(s_model, g1_model, g2_model, s_data, g1_data, g2_data)
println("\nDone! =]\n")
