function truncate_to_square(mat1::Matrix{T}, mat2::Matrix{T}) where {T}
    # Step 1: Get the size of the matrices
    size1 = size(mat1)
    size2 = size(mat2)
    
    # Step 2: Determine the minimum dimension
    min_dim = min(size1[1], size1[2], size2[1], size2[2])
    
    # Step 3: Create new square matrices
    square_mat1 = Matrix{T}(undef, min_dim, min_dim)
    square_mat2 = Matrix{T}(undef, min_dim, min_dim)
    
    # Step 4: Assign corresponding elements
    for i in 1:min_dim, j in 1:min_dim
        square_mat1[i, j] = mat1[i, j]
        square_mat2[i, j] = mat2[i, j]
    end
    
    return square_mat1, square_mat2
end


function meshgrid(x, y)
    X = [i for i in x, j in 1:length(y)]
    Y = [j for i in 1:length(x), j in y]
    return X, Y
end

function ks(g1Map, g2Map; meshgrid = meshgrid, ts = truncate_to_square)
  g1Map, g2Map = ts(g1Map, g2Map)
  x, y = size(g1Map)
  k1, k2 = meshgrid(fftfreq(y), fftfreq(x))

  g1_hat = fft(g1Map)
  g2_hat = fft(g2Map)

  p1 = k1 * k1 - k2 * k2
  p2 = 2 * k1 * k2
  k2 = k1 * k1 + k2 * k2
  k2[1,1] = 1

  kEhat = (p1 * g1_hat + p2 * g2_hat) / k2
  kBhat = -(p2 * g1_hat - p1 * g2_hat) / k2

  kE = real.(ifft(kEhat))
  kB = real.(ifft(kBhat))

  return kE, kB
end
