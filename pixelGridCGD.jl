function pgCost(params)
  Totalcost = 0
  for i in 1:length(params)
    Totalcost += 0.5*(params[i] - vec(star)[i])^2
  end
  return Totalcost
end

function pg_g!(storage, p)
  grad_cost = ForwardDiff.gradient(pgCost, p)
  for i in 1:(r*c)                                  
    storage[i] = grad_cost[i]                      
  end                                                                                                 
end
