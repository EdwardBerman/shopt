include("radialProfiles.jl")

function cost(params) 
  Totalcost = 0
  #print(params)
  σ = params[1]
  s_guess = σ^2
  e1_guess = params[2]
  e2_guess = params[3]
  ellipticity = sqrt((e1_guess)^2 + (e2_guess)^2)
  normG = sqrt(1 + 0.5*( (1/ellipticity^2) - sqrt( (4/ellipticity^2)+ (1/ellipticity^4)  )  ))
  ratio = ellipticity/normG
  g1_guess = e1_guess/ratio
  g2_guess = e2_guess/ratio
 
  starGuess = zeros(10,10)
  starGuess[5,5] = 1
  starGuess[6,5] = 1
  starGuess[5,6] = 1
  starGuess[6,6] = 1

  normGuess = zeros(10,10)
  normGuess[5,5] = 1
  normGuess[6,5] = 1
  normGuess[5,6] = 1
  normGuess[6,6] = 1
  

  for u in 1:10
    for v in 1:10
      normGuess[u,v] = EGaussian(1, u, v, g1, g2, s)
    end
  end
  A_guess = 1/sum(normGuess)

  for u in 1:10
    for v in 1:10
      Totalcost += 0.5*(EGaussian(A_guess, u, v, g1_guess, g2_guess, s_guess) - star[u,v])^2
    end
  end
  return Totalcost
end


function g!(storage, p)
    storage[1] = ForwardDiff.gradient(cost, p)[1]
    storage[2] = ForwardDiff.gradient(cost, p)[2]
    storage[3] = ForwardDiff.gradient(cost, p)[3]
end

