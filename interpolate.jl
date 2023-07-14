#=
Helper functions to interpolate s(u,v), g1(u,v), g2(u,v) across the field of view
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

#=
I = optimize(interpCost, polyG!, rand(10), ConjugateGradient())
learnedPolynomial = zeros(10,10)
IC = I.minimizer

for i in 1:10
  for j in 1:10
    learnedPolynomial[i,j] = IC[1]*i^3 + IC[2]*j^3 + IC[3]*i^2*j + IC[4]*j^2*i + IC[5]*i^2 + IC[6]*j^2 + IC[7]*i*j + IC[8]*i + IC[9]*j + IC[10]
  end
end

plot(heatmap(my_truth), heatmap(learnedPolynomial), heatmap(my_truth - learnedPolynomial))
=#
