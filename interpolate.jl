#=
Helper functions to interpolate s(u,v), g1(u,v), g2(u,v) across the field of view
Reminder to make s, g1, g2 (u,v) of arbitrary degree
=#

#f(u, v) = a1u^3 + a2v^3 + a3u^2v + a4v^2u + a5u^2 + a6v^2 + a7uv + a8u + a9v + a10
function interpCostS(p; truth=s_tuples)
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

function polyG_s!(storage, p)
  grad_cost = ForwardDiff.gradient(interpCostS, p) #∇
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

function interpCostg1(p; truth=g1_tuples)
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

function polyG_g1!(storage, p)
  grad_cost = ForwardDiff.gradient(interpCostg1, p)
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

function interpCostg2(p; truth=g2_tuples)

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

function polyG_g2!(storage, p)
  grad_cost = ForwardDiff.gradient(interpCostg2, p)
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
    loss = sum((objective_function(p, x_val, y_val, degree) - z_actual)^2  for ((x_val, y_val), z_actual) in zip(zip(x_data, y_data), z_data) if !isnan(z_actual))
    return loss
  end 
  result = optimize(objective, initial_guess, autodiff=:forward, LBFGS(), Optim.Options(g_tol = polynomial_interpolation_stopping_gradient, f_tol=1e-40)) #autodiff=:forward 
  return Optim.minimizer(result)
end

