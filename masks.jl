#=
Masks for NaN Handeling, used in dataPreprocessing and for Pixel Grid Fitting 
=#

function nanMask(arr)
  dummyArr = zeros(size(arr,1),size(arr,2))
  for i in 1:size(arr,1)
    for j in 1:size(arr,2)
      if arr[i,j] < -1000
        dummyArr[i,j] = NaN
      else
        dummyArr[i,j] = arr[i,j]
      end
    end
  end
  return dummyArr
end

#For Purposes of Summing to unity
function nanToGaussian(arr, s, g1, g2, uc, vc, func=fGaussian)
  dummyArr = zeros(size(arr,1),size(arr,2))
  for i in 1:size(arr,1)
    for j in 1:size(arr,2)
      if isnan(arr[i,j])
        mu = 0.0
        sigma = 1
        normal_dist = Normal(mu, sigma)
        truncated_dist = Truncated(normal_dist, -0.1*func(i, j, s, g1, g2, uc, vc), 0.1*func(i, j, s, g1, g2, uc, vc))
        dummyArr[i,j] = func(i, j, s, g1, g2, uc, vc) + rand(truncated_dist)
      else
        dummyArr[i,j] = arr[i,j]
      end
    end
  end
  return dummyArr
end

function nanToZero(arr)
  dummyArr = zeros(size(arr,1),size(arr,2))
  for i in 1:size(arr,1)
    for j in 1:size(arr,2)
      if isnan(arr[i,j])
        dummyArr[i,j] = 0
      else
        dummyArr[i,j] = arr[i,j]
      end
    end
  end
  return dummyArr
end
function nanToInf(arr)
  dummyArr = zeros(size(arr,1),size(arr,2))
  for i in 1:size(arr,1)
    for j in 1:size(arr,2)
      if isnan(arr[i,j])
        dummyArr[i,j] = -1*Inf
      else
        dummyArr[i,j] = arr[i,j]
      end
    end
  end
  return dummyArr
end

function nanMask2(arr)
  dummyArr = zeros(size(arr,1),size(arr,2))
  for i in 1:size(arr,1)
    for j in 1:size(arr,2)
      if arr[i,j] < 0
        dummyArr[i,j] = NaN
      else
        dummyArr[i,j] = arr[i,j]
      end
    end
  end
  return dummyArr
end

#note to self add nan to sky background
