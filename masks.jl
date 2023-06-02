function nanMask(arr)
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

#For Purposes of Summing to unity
function nanMask(arr)
  dummyArr = zeros(size(arr,1),size(arr,2))
  for i in 1:size(arr,1)
    for j in 1:size(arr,2)
      if arr[i,j] < 0
        dummyArr[i,j] = 0
      else
        dummyArr[i,j] = arr[i,j]
      end
    end
  end
  return dummyArr
end

