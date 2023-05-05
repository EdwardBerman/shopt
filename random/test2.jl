using Optim

it = []
fv = []
norm = []
# Returns a closure over a logger, that takes a Optim trace as input
function cb(opt_state:: Optim.OptimizationState)
  push!(it, opt_state.iteration)
  push!(fv, opt_state.value)
  push!(norm, opt_state.g_norm)
  return false  # do not terminate optimisation
end


rosenbrock(x) =  (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
result = optimize(rosenbrock, zeros(2), BFGS(), Optim.Options(callback=cb))#callback=cb
print("\n \n", it, "\n \n", fv, "\n \n", norm, "\n \n", it[1], "\n \n \n")
