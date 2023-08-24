function chisq_cost(params; starL=starCatalog[iteration], weight_map=errVignets[iteration])    
  pixel_values = params
  chisq = sum(x -> isfinite(x) ? x : 0, (pixel_values .- vec(starL)) .^ 2 ./ vec(weight_map))
  return chisq 
end 


function chisq_g!(storage, p)
  chisq_grad_cost = ForwardDiff.gradient(chisq_cost, p)
  storage[1:length(chisq_grad_cost)] = chisq_grad_cost
end

