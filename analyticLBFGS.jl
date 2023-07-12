include("radialProfiles.jl")

#=
Functions for Cost and Gradient used in the Optimize step with LBFGS
NB: Reparameterization for [s, g1, g2] via [σ, e1, e2] to constraint update steps inside R+ x B_2(r)
=#

function cost(params; r = r, c= c, starL=starCatalog[iteration], radial=fGaussian, AnalyticStampSize=AnalyticStampSize, get_middle_nxn=get_middle_nxn) 
  Totalcost = 0
  σ = params[1]
  s_guess = σ^2
  e1_guess = params[2]
  e2_guess = params[3]
  ellipticity = sqrt((e1_guess)^2 + (e2_guess)^2)
  normG = sqrt(1 + 0.5*( (1/ellipticity^2) - sqrt( (4/ellipticity^2)+ (1/ellipticity^4)  )  ))
  ratio = ellipticity/normG
  g1_guess = e1_guess/ratio
  g2_guess = e2_guess/ratio
 
  starL = get_middle_nxn(starL, AnalyticStampSize)
  r = AnalyticStampSize
  c = AnalyticStampSize

  sum = 0
  for u in 1:r
    for v in 1:c
      try
        sum +=  radial(u,v, g1_guess, g2_guess, s_guess, r/2,c/2)
      catch
        sum += 0
      end
    end
  end
  A_guess = 1/sum
  
  for u in 1:r
    for v in 1:c
      if isnan(starL[u,v])
        Totalcost += 0
      else
        Totalcost += 0.5*(A_guess*radial(u, v, g1_guess, g2_guess, s_guess, r/2, c/2) - starL[u, v])^2
      end
    end
  end
  return Totalcost
end


function costD(params; r=r, c=c, starL=pixelGridFits[iteration], radial=fGaussian, AnalyticStampSize=AnalyticStampSize, get_middle_nxn=get_middle_nxn) 
  Totalcost = 0
  σ = params[1]
  s_guess = σ^2
  e1_guess = params[2]
  e2_guess = params[3]
  ellipticity = sqrt((e1_guess)^2 + (e2_guess)^2)
  normG = sqrt(1 + 0.5*( (1/ellipticity^2) - sqrt( (4/ellipticity^2)+ (1/ellipticity^4)  )  ))
  ratio = ellipticity/normG
  g1_guess = e1_guess/ratio
  g2_guess = e2_guess/ratio
  
  starL = get_middle_nxn(starL, AnalyticStampSize)
  r = AnalyticStampSize
  c = AnalyticStampSize

  sum = 0
  for u in 1:r
    for v in 1:c
      try
        sum +=  radial(u,v, g1_guess, g2_guess, s_guess, r/2,c/2)
      catch
        sum += 0
      end
    end
  end
  A_guess = 1/sum
  
  for u in 1:r
    for v in 1:c
      if isnan(starL[u,v])
        Totalcost += 0
      else
        Totalcost += 0.5*(A_guess*radial(u, v, g1_guess, g2_guess, s_guess, r/2, c/2) - starL[u, v])^2
      end
    end
  end
  return Totalcost
end

function g!(storage, p)
    grad_cost = ForwardDiff.gradient(cost, p)
    storage[1] = grad_cost[1]
    storage[2] = grad_cost[2]
    storage[3] = grad_cost[3]
end

function gD!(storage, p)
  grad_cost = ForwardDiff.gradient(costD, p)
  storage[1] = grad_cost[1]
  storage[2] = grad_cost[2]
  storage[3] = grad_cost[3]
end
