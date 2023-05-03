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
  storage[1] = ForwardDiff.gradient(interpCost, p)[1]
  storage[2] = ForwardDiff.gradient(interpCost, p)[2]
  storage[3] = ForwardDiff.gradient(interpCost, p)[3]
  storage[5] = ForwardDiff.gradient(interpCost, p)[4]
  storage[5] = ForwardDiff.gradient(interpCost, p)[5]
  storage[6] = ForwardDiff.gradient(interpCost, p)[6]
  storage[7] = ForwardDiff.gradient(interpCost, p)[7]
  storage[8] = ForwardDiff.gradient(interpCost, p)[8]
  storage[9] = ForwardDiff.gradient(interpCost, p)[9]
  storage[10] = ForwardDiff.gradient(interpCost, p)[10]
end

