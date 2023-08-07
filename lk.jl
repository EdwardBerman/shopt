# Defining Lanczos kernel
function Ln(x, n)
    if x == 0
        return 1
    elseif abs(x) < n
        return (sin(π*x)*sin(π*x/n))/(π^2*n*x^2)
    else
        return 0
    end
end

# Defining function I
function I(u, v, p, u_k, v_k, n)
    return sum([p[k]*Ln(u-u_k[k], n)*Ln(v-v_k[k], n) for k in 1:length(p)])
end

# Creating model grid
Npix = r*c # choose the size
u_k = range(-1, stop=1, length=r)
v_k = range(-1, stop=1, length=c)
u_k, v_k = vec([i for i in u_k, j in v_k]), vec([j for i in u_k, j in v_k]) # grid centers

# Initial guess for p_k
p = rand(r*c)

# Optimization function to minimize
function lk_loss(p; true_image=vec(starCatalog[iteration]))
    loss_val = 0.0
    for (idx, u) in enumerate(u_k)
        v = v_k[idx]
        loss_val += (I(u, v, p, u_k, v_k, 3) - true_image[idx])^2 / true_image[idk] # assuming n=3 for the kernel function
    end
    return loss_val
end

function ∇lk!(storage, p)
  grad_I = ForwardDiff.gradient(lk_loss, p)
  for i in 1:length(p)
    storage[i] = grad_I[i]
  end
end

