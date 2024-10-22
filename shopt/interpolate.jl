#=
Helper functions to interpolate s(u,v), g1(u,v), g2(u,v) across the field of view
Reminder to make s, g1, g2 (u,v) of arbitrary degree!
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
        value += p[counter] * x^(i - 1) * y^(j - 1) 
      end
    end
  end 
  return value    
end


function polynomial_optimizer(degree, x_data, y_data, z_data; median_constraint=median_constraint)
    function generate_polynomial_terms(n)
        terms = []
        for i in 0:n
            for j in 0:(n-i)
                push!(terms, (i, j))
            end
        end
        terms
    end
    polynomial_terms = generate_polynomial_terms(degree)
    A = zeros(length(z_data), length(polynomial_terms))
    #replace nans with median 
    if median_constraint == true
        A[:, 1] .= median(filter(!isnan, z_data))
        for i in 1:length(z_data)
            for (index, (p, q)) in enumerate(polynomial_terms[2:end])  # Skip the constant term
                A[i, index+1] = x_data[i]^p * y_data[i]^q
            end
        end
    else 
        for i in 1:length(z_data)
            if isnan(z_data[i])
                z_data[i] = median(filter(!isnan, z_data))
            end
        end
        for i in 1:length(z_data)
            for (index, (p, q)) in enumerate(polynomial_terms)
                A[i, index] = x_data[i]^p * y_data[i]^q
            end
        end
    end

    coefficients = A \ z_data
    if median_constraint == true
        coefficients[1] = median(filter(!isnan, z_data))
    end
    if length(z_data) < 5
        coefficients = zeros(length(coefficients))
        coefficients[1] = median(filter(!isnan, z_data))
    end
    return coefficients
end

# if statement that makes a new polynomial_optimizer function that has a median constraint. Objective function will only be a function of p-1 terms, will add median(z_data) for constant term
