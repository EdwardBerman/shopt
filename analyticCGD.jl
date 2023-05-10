include("radialProfiles.jl")

function cost(params) 
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
  

  normGuess = zeros(r,c)
  
  if isodd(r) & isodd(c)
    normGuess[(r÷2) + 1, (c÷2)+1] = 1
  end
  if iseven(r) & iseven(c)
    normGuess[r÷2, c÷2] = 1
    normGuess[(r÷2) + 1, c÷2] = 1
    normGuess[r÷2, (c÷2) + 1] = 1
    normGuess[(r÷2) + 1, (c÷2) + 1] = 1
  end
  if isodd(r) & iseven(c)
    normGuess[(r÷2) + 1, c÷2] = 1
    normGuess[(r÷2) + 1, (c÷2) + 1] = 1
  end
  if iseven(r) & isodd(c)
    normGuess[median([1:1:r;]) - 0.5, median([1:1:c;])] = 1
    normGuess[median([1:1:r;]) + 0.5, median([1:1:c;])] = 1
  end
  try
    sum = 0
    for u in 1:r
      for v in 1:c
        sum +=  EGaussian(u,v, g1_guess, g2_guess, s_guess, r/2,c/2)
      end
    end
    A_guess = 1/sum
    for u in 1:r
      for v in 1:c
        Totalcost +=0.5*( A_guess*EGaussian(u,v, g1_guess, g2_guess, s_guess, r/2,c/2) - star[u,v])^2
      end
    end
  catch MethodError
    Totalcost = 0
  end
  return Totalcost
end


function g!(storage, p)
    grad_cost = ForwardDiff.gradient(cost, p)
    storage[1] = grad_cost[1]
    storage[2] = grad_cost[2]
    storage[3] = grad_cost[3]
end

