function detect_outliers(data::AbstractVector{T}; k::Float64=1.5) where T<:Real
  q1 = quantile(data, 0.25)
  q3 = quantile(data, 0.75)
  iqr = q3 - q1
  lower_fence = q1 - k * iqr 
  upper_fence = q3 + k * iqr
  filter = (data .< lower_fence) .| (data .> upper_fence)
  outliers = data[filter]
  return outliers
end

function outliers_filter(snr::Vector{Any}, img::Vector{Any}, wht::Vector{Matrix{Float32}}, k::Float64) 
  q1 = quantile(snr, k)
  println("━ Cutting off Stars below the $k Percentile of Signal to Noise Ratio: $q1 , based off of snr = 10log[Σpix(I²/σ²)]")
  q3 = quantile(snr, 0.75)
  iqr = q3 - q1
  lower_fence = q1 - k * iqr 
  upper_fence = q3 + k * iqr
  outlier_indices = findall(x -> x < q1, snr) #outlier_indices = findall(x -> x < lower_fence || x > upper_fence, snr)
  img_snr_cleaned = img
  wht_snr_cleaned = wht
  for i in sort(outlier_indices, rev=true)
    splice!(img_snr_cleaned, i)
    splice!(wht_snr_cleaned, i)
  end
  return img_snr_cleaned, wht_snr_cleaned
end

function remove_outliers(data::AbstractVector{T}; k::Float64=1.5) where T<:Real
  q1 = quantile(data, 0.25)
  q3 = quantile(data, 0.75)
  iqr = q3 - q1
  lower_fence = q1 - k * iqr
  upper_fence = q3 + k * iqr
  filter = (data .> lower_fence) .& (data .< upper_fence)
  nonOutliers = data[filter]
  return nonOutliers
end

