function EGaussian(u, v, g1, g2, s, uc, vc)
  squareRadiusFactor = ([(1 + g1) g2; g2 (1 - g1)]*[(u - uc); (v- vc)])'*([(1 + g1) g2; g2 (1 - g1)]*[(u-uc);(v-vc)])
  matrixScalar = s^2/(1 - g1^2 - g2^2)
  return exp(-(matrixScalar*squareRadiusFactor))
end

function Fkolmogorov(A, k, λ)
  r0 = 1
  λ = 1
  return A*exp(-(24*gamma(6.5)*0.2)^(5/6) * (λ*k/(2*π*r0)^(5/3)))
end

function fkolmogorov(A, λ, u, v, g1, g2, s, Fk=Fkolmogorov)
  squareRadiusFactor = ([(1 + g1) g2; g2 (1 - g1)]*[(u - 5.5); (v- 5.5)])'*([(1 + g1) g2; g2 (1 - g1)]*[(u-5.5);(v-5.5)])
  matrixScalar = s^2/(1 - g1^2 - g2^2)
  squareRadius = squareRadiusFactor*matrixScalar
  integrand(k)  = Fk(A, k, λ) * besselj(0, k*(squareRadius^(0.5)) )* k
  result, err = quadgk(integrand, 0, Inf)
  return (1/(2*π)) * result
end



