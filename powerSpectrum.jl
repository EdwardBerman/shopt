#=
Computers the Power Spectra at a given annulus, called iteratively for a Power Spectra map
=#

function powerSpectrum(data::Array{T, 2}, radius) where T<:Real
  radiusPixels = []
  for u in 1:size(data,1)    
    for v in 1:size(data,2)   
      if round(sqrt((u - size(data,1)/2)^2 + (v - size(data,2)/2)^2) - radius) == 0 1 
        push!(radiusPixels, data[u,v])
      end
    end
  end
  return mean(radiusPixels)

end
