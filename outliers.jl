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

