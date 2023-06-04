#f(u, v) = a1u^3 + a2v^3 + a3u^2v + a4v^2u + a5u^2 + a6v^2 + a7uv + a8u + a9v + a10
#Just use optim again

#using Optim
#using Plots
#using ForwardDiff

my_truth = zeros(10,10)

for i in 1:10
  for j in 1:10
    my_truth[i,j] = (i)^2 + (j)^2
  end
end

#plot(heatmap(my_truth))

function convert_array_to_tuples(arr)
    tuples_list = []
    for i in 1:size(arr, 1)
        for j in 1:size(arr, 2)
            push!(tuples_list, (i, j, arr[i, j]))
        end
    end
    return tuples_list
end

my_truth_new = convert_array_to_tuples(my_truth)

function interpCost(p; truth=my_truth_new)
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
