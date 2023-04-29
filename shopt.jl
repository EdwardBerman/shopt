using Base: initarray!
using BenchmarkTools
using Manopt
using ManifoldsBase
using Manifolds
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

include("plot.jl")
include("optArgs.jl")
include("radialProfiles.jl")

print("Insert [s, g1, g2]")

print("\n s = ")
s = readline()
s = parse(Float64, s)

print("\n g1 = ")
g1 = readline()
g1 = parse(Float64, g1)

print("\n g2 = ")
g2 = readline()
g2 = parse(Float64, g2)

print("\n noise = ")
noise = readline()
noise = parse(Float64, noise)

star = zeros(10, 10)
star[5,5] = 1
star[5,6] = 1
star[6,5] = 1
star[6,6] = 1

norm = zeros(10, 10)
norm[5,5] = 1
norm[5,6] = 1
norm[6,5] = 1
norm[6,6] = 1

for u in 1:10
  for v in 1:10
    norm[u,v] = EGaussian(1, u, v, g1, g2, s)
  end
end

A = 1/sum(norm)
print("A = ", A, "\n")

for u in 1:10
  for v in 1:10
    star[u,v] = EGaussian(A, u, v, g1, g2, s)
    star[u,v] = star[u,v] + rand(Normal(-noise*star[u,v], star[u,v]*noise))
  end
end


starData = zeros(10, 10)
starData[5,5] = 1
starData[5,6] = 1
starData[6,5] = 1
starData[6,6] = 1


shearManifold = Euclidean(3)

#stopping_criterion = StopWhenAny(StopAfterIteration(10000),StopWhenGradientNormLess(1e-12))
#stopping_criterion = StopWhenGradientNormLess(1e-12)
#stopping_criterion = StopAfterIteration(1000)

itr = 20
A_data = zeros(itr)
s_data = zeros(itr)
g1_data = zeros(itr)
g2_data = zeros(itr)
e1_data = zeros(itr)
e2_data = zeros(itr)

@time begin
  for i in 1:itr
    initial_guess = rand(shearManifold)
    print("\n initial guess [s, e1, e2] \n", initial_guess)
    
    x_cg = optimize(cost, g!, initial_guess, ConjugateGradient())
    println(x_cg)
    s_data[i] = x_cg.minimizer[1]^2
    e1_guess = x_cg.minimizer[2]
    e2_guess = x_cg.minimizer[3]

    ellipticityData = sqrt((e1_guess)^2 + (e2_guess)^2)
    normGdata = sqrt(1 + 0.5*( (1/ellipticityData^2) - sqrt( (4/ellipticityData^2) + (1/ellipticityData^4)  )  )) 
    ratioData = ellipticityData/normGdata
    g1_data[i] = e1_guess/ratioData            
    g2_data[i] = e2_guess/ratioData  
    
    norm_data = zeros(10,10)
    norm_data[5,5] = 1
    norm_data[5,6] = 1
    norm_data[6,5] = 1
    norm_data[6,6] = 1

    for u in 1:10
      for v in 1:10
        norm_data[u,v] = EGaussian(1, u, v, g1_data[i], g2_data[i], s_data[i])
      end
    end
    A_data[i] = 1/sum(norm_data)
    print("\n", "A: ", A_data[i], "   s: ", s_data[i]^2, "   g1: ", g1_data[i], "   g2: ", g2_data[i], "\n \n \n")

    temp = zeros(10, 10)
    for u in 1:10
      for v in 1:10
        temp[u,v] = EGaussian(A_data[i], u, v, g1_data[i], g2_data[i], s_data[i])
      end
    end
  end
end

for j in 1:itr
  print("\n    A: ", A_data[j], "\n   s: ", s_data[j], "\n   g1: ", g1_data[j], "\n   g2: ", g2_data[j], "\n \n \n" )
end

function detect_outliers(data::AbstractVector{T}; k::Float64=1.5) where T<:Real
    q1 = quantile(data, 0.25)
    q3 = quantile(data, 0.75)
    iqr = q3 - q1
    lower_fence = q1 - k * iqr
    upper_fence = q3 + k * iqr
    filter = (data .< lower_fence) .| (data .> upper_fence)
    outliers = data[filter]
    return outliers
end

print("Outliers in s: ", detect_outliers(s_data), "\n")
ns = length(detect_outliers(s_data))
ng1 = length(detect_outliers(g1_data))
ng2 = length(detect_outliers(g2_data))

print("Number of outliers in s: ", ns[1], "\n")
print("Number of outliers in g1: ", ng1[1], "\n")
print("Number of outliers in g2: ", ng2[1], "\n")

function remove_outliers(data::AbstractVector{T}; k::Float64=1.5) where T<:Real
    q1 = quantile(data, 0.25)
    q3 = quantile(data, 0.75)
    iqr = q3 - q1
    lower_fence = q1 - k * iqr
    upper_fence = q3 + k * iqr
    filter = (data .> lower_fence) .& (data .< upper_fence)
    nonOutliers = data[filter]
    return nonOutliers
end

#Plotting
error_plot([s, g1, g2], [mean(s_data), mean(g1_data), mean(g2_data)], [std(s_data)/sqrt(itr), std(g1_data)/sqrt(itr), std(g2_data)/sqrt(itr)], "Learned vs True Parameters")

s_data = remove_outliers(s_data)
g1_data = remove_outliers(g1_data)
g2_data = remove_outliers(g2_data)

norm2 = zeros(10, 10)
norm2[5,5] = 1
norm2[5,6] = 1
norm2[6,5] = 1
norm2[6,6] = 1

for u in 1:10
  for v in 1:10
    norm2[u,v] = EGaussian(1, u, v, mean(g1_data), mean(g2_data), mean(s_data))
  end
end

A_data = 1/sum(norm2)

for u in 1:10
  for v in 1:10
    starData[u,v] = EGaussian(A_data, u, v, mean(g1_data), mean(g2_data), mean(s_data))
  end
end

Residuals = star - starData
costSquaredError = Residuals.^2 
chiSquare = zeros(10, 10)
for u in 1:10
  for v in 1:10
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

plot_all()

ns = size(s_data, 1)
ng1 = size(g1_data, 1)
ng2 = size(g2_data, 1)

error_plot([s, g1, g2], [mean(s_data), mean(g1_data), mean(g2_data)], [std(s_data)/sqrt(ns), std(g1_data)/sqrt(ng1), std(g2_data)/sqrt(ng2)], "Learned vs True Parameters Outliers Removed")

dof = 78 #9 x 9 x 3 - 3
p = ccdf(Chisq(dof), sum(chiSquare))
#print(Chisq(dof), sum(chiSquare), "\n")
print("p-value \n", p, "\n")