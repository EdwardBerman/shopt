include("radialProfiles.jl")
function cost(::AbstractManifold, params) 
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
      Totalcost += 0.5*(EGaussian(A_guess, u, v, g1_guess, g2_guess, s_guess) - EGaussian(A, u, v, g1, g2, s))^2
    end
  end
  return [Totalcost]
end

function jacF_RLM(M::AbstractManifold, p; basis_domain::AbstractBasis=DefaultOrthonormalBasis())
  #X0 = zeros(manifold_dimension(M))
  J = ForwardDiff.jacobian(
        #x -> cost(M, p), p
        x -> cost(M, exp(M, p, get_vector(M, p, x, basis_domain))), p
  )
  return J
end

function cost(params; radialProfile=EGaussian) 
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
      normGuess[u,v] = radialProfile(1, u, v, g1, g2, s)
    end
  end
  A_guess = 1/sum(normGuess)

  for u in 1:10
    for v in 1:10
      Totalcost += 0.5*(radialProfile(A_guess, u, v, g1_guess, g2_guess, s_guess) - radialProfile(A, u, v, g1, g2, s))^2
    end
  end
  return Totalcost
end

#grad_f(::Euclidean, p) = ForwardDiff.gradient(cost2, p)
grad_f(p) = ForwardDiff.gradient(cost2, p)

function cgd(f, df, x0, max_iters=1000, tol=1e-3)
  # Initialize variables
  x = x0
  grad = df(x)
  d = -grad
  delta_new = grad' * grad
  delta_0 = delta_new
  # Perform conjugate gradient descent
  for i in 1:max_iters
    # Compute step size
    alpha = delta_new / (d' * df(d))
    # Update x and grad
    x = x + alpha * d
    grad_new = df(x)
    delta_old = delta_new
    delta_new = grad_new' * grad_new
                     
    # Check for convergence
    if delta_new < tol^2 * delta_0
      break
    end
                      
    # Update conjugate direction
    beta = delta_new / delta_old
    d = -grad_new + beta * d
                      
    # Update gradient
    grad = grad_new
  end
                 
  print(x)                
  return x
end
function g!(storage, p)
    storage[1] = ForwardDiff.gradient(cost, p)[1]
    storage[2] = ForwardDiff.gradient(cost, p)[2]
    storage[3] = ForwardDiff.gradient(cost, p)[3]
end

