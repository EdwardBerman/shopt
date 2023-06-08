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
