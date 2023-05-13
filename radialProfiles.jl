function fGaussian(u, v, g1, g2, s, uc, vc)
  squareRadiusFactor = ([(1 + g1) g2; g2 (1 - g1)]*[(u - uc); (v- vc)])'*([(1 + g1) g2; g2 (1 - g1)]*[(u-uc);(v-vc)])
  matrixScalar = s^2/(1 - g1^2 - g2^2)
  return exp(-(matrixScalar*squareRadiusFactor))
end

function Fkolmogorov(k, λ)
  r0 = 1
  return exp(-(24*gamma(6.5)*0.2)^(5/6) * (λ*k/(2*π*r0)^(5/3)))
end

function fkolmogorov(u, v, g1, g2, s, uc, vc, Fk=Fkolmogorov)
  λ = 1
  squareRadiusFactor = ([(1 + g1) g2; g2 (1 - g1)]*[(u - uc); (v- vc)])'*([(1 + g1) g2; g2 (1 - g1)]*[(u-uc);(v-vc)])
  matrixScalar = s^2/(1 - g1^2 - g2^2)
  squareRadius = squareRadiusFactor*matrixScalar
  integrand(k)  = Fk(k, λ) * besselj(0, k*(squareRadius^(0.5)) )* k
  result, err = quadgk(integrand, 0, Inf)
  return (1/(2*π)) * result
end



