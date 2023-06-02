#f(u, v) = a1u^3 + a2v^3 + a3u^2v + a4v^2u + a5u^2 + a6v^2 + a7uv + a8u + a9v + a10
#Just use optim again


function interpCost(p, truth)
  f(u, v) = p[1]*u^3 + p[2]*v^3 + p[3]*u^2*v + p[4]*v^2*u + p[5]*u^2 + p[6]*v^2 + p[7]*u*v + p[8]*u + p[9]*v + p[10]
  t = truth
  function sumLoss(func=f, t)
    totalLoss = 0
    for i in 1:length(t)
      totalLoss += (func(u[i], v[i]) - t[u,v])^2
    end
    return totalLoss
  end
  return sumLoss(f, t)
end

interpCost = (p) -> interpCost(p, truth) #truth = pik


function polyG!(storage, p)
  grad_cost = ForwardDiff.gradient(interpCost, p)
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

