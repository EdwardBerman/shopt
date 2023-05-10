function pgCost(params)
  Totalcost = 0
  for i in 1:length(params)
    Totalcost += 0.5*(params[i] - vec(star)[i])^2
  end
  return Totalcost
end


function pg_g!(storage, p)
  storage = storage
  for i in 1:100
    storage[i] = ForwardDiff.gradient(pgCost, p)[i]
  end
end

