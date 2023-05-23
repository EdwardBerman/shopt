function pgCost(params; starL=starCatalog[iteration])
  Totalcost = 0
  for i in 1:length(params)
    a = 0
    b = 0
    if params[i] > 0
      a = params[i]
    else
      a = 0.000001
    end
    if vec(starL)[i] > 0
      b = vec(starL)[i]
    else
      b = 0.000001
    end

    Totalcost += 0.5*(log(a) - log(b))^2
  end
  return Totalcost
end

function pg_g!(storage, p)
  grad_cost = ForwardDiff.gradient(pgCost, p)
  for i in 1:(r*c)                                  
    storage[i] = grad_cost[i]                      
  end                                                                                                 
end
