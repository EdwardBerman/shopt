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
using Manifolds
using ManifoldsBase

include("plot.jl")
include("analyticCGD.jl")
include("radialProfiles.jl")
include("pixelGridCGD.jl")
include("dataPreprocessing.jl")
include("outliers.jl")

println("Processing Data for Fit")
star, r, c = dataprocessing()

for u in 1:r
  for v in 1:c
    star[u,v] = star[u,v] + rand(Normal(-0.1*star[u,v], star[u,v]*0.1))
  end
end

 
starData = zeros(r, c)
shearManifold = Euclidean(3)

itr = 5
A_data = zeros(itr)
s_data = zeros(itr)
g1_data = zeros(itr)
g2_data = zeros(itr)


ltPlots = []

println("Analytic Profile Fit for Model Star")
@time begin
  for i in 1:itr
    println("\t Iteration $i")
    initial_guess = rand(shearManifold)
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
                    Optim.Options(callback = cb))#store_trace=true

    loss_time = plot(it, 
                     loss, 
                     title="Analytic Profile Loss Vs Iteration", 
                     xlabel="Iteration", 
                     ylabel="Loss",
                     label="Star $itr Model")
    push!(ltPlots, loss_time)
    #=
    if "$i" == "20"
      savefig(ltPlots, joinpath("outdir", "lossTime.png")) 
    end
    =#


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
        norm_data[u,v] = EGaussian(u, v, g1_data[i], g2_data[i], s_data[i], r/2, c/2)
      end
    end
    A_data[i] = 1/sum(norm_data)
    println("\t Found A: ", A_data[i], "\t s: ", s_data[i]^2, "\t g1: ", g1_data[i], "\t g2: ", g2_data[i])

  end
end

println("\t \t Outliers in s: ", detect_outliers(s_data))
ns = length(detect_outliers(s_data))
ng1 = length(detect_outliers(g1_data))
ng2 = length(detect_outliers(g2_data))

println("\t \t Number of outliers in s: ", ns[1])
println("\t \t Number of outliers in g1: ", ng1[1])
println("\t \t Number of outliers in g2: ", ng2[1])

println("Pixel Grid Fit")
pg = optimize(pgCost, pg_g!, zeros(r*c), ConjugateGradient())
print(pg)
pg = reshape(pg.minimizer, (r, c))


#Plotting Error True Vs Learned
###error_plot([s, g1, g2], [mean(s_data), mean(g1_data), mean(g2_data)], [std(s_data)/sqrt(itr), std(g1_data)/sqrt(itr), std(g2_data)/sqrt(itr)], "Learned vs True Parameters")

# Plotting Heatmaps
print("\nPlotting \n")
s_data = remove_outliers(s_data)
g1_data = remove_outliers(g1_data)
g2_data = remove_outliers(g2_data)

hist(g1_data, g2_data)

norm2 = zeros(r, c)
norm2[5,5] = 1
norm2[5,6] = 1
norm2[6,5] = 1
norm2[6,6] = 1

for u in 1:r
  for v in 1:c
    norm2[u,v] = EGaussian(u, v, mean(g1_data), mean(g2_data), mean(s_data), r/2, c/2)
  end
end
A_data = 1/sum(norm2)

for u in 1:r
  for v in 1:c
    starData[u,v] = A_data*norm2[u,v]
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

plot_hm()

ns = size(s_data, 1)
ng1 = size(g1_data, 1)
ng2 = size(g2_data, 1)

###error_plot([s, g1, g2], [mean(s_data), mean(g1_data), mean(g2_data)], [std(s_data)/sqrt(ns), std(g1_data)/sqrt(ng1), std(g2_data)/sqrt(ng2)], "Learned vs True Parameters Outliers Removed")

dof = r*c - 3
p = 1 - cdf(Chisq(dof), sum(chiSquare))#ccdf = 1 - cdf
#print(Chisq(dof), sum(chiSquare), "\n")
print("p-value: ", p, "\n")


