#=
Functions for detecting and removing outliers from data, used in dataPreprocessing.jl
=#

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

function outliers_filter(snr::Vector{Float32}, img::Vector{Any}, k::Float64) 
  #snr_cleaned = filter(!isnan, snr)
  q1 = quantile(snr, k)
  #snr = [isnan(x) ? -1000 : x for x in snr]
  println("━ Cutting off Stars below the $k Percentile of Signal to Noise Ratio: $q1 ")
  q3 = quantile(snr, 0.75)
  iqr = q3 - q1
  lower_fence = q1 - k * iqr 
  upper_fence = q3 + k * iqr
  outlier_indices = findall(x -> x < q1, snr) #outlier_indices = findall(x -> x < lower_fence || x > upper_fence, snr)
  println("━ outlier indices: $outlier_indices")
  img_snr_cleaned = img
  #wht_snr_cleaned = wht
  for i in sort(outlier_indices, rev=true)
    splice!(img_snr_cleaned, i)
    #splice!(wht_snr_cleaned, i)
  end
  return img_snr_cleaned, outlier_indices
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

