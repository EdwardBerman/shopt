using PyCall

#=
Functions for reading in summary.shopt file and returning the PSF at an arbitrary (u,v)
=#

#=
Example:
polyMatrix, deg = read_shopt("summary.shopt")
p(0.5,0.5,polyMatrix, deg)
=#

function read_shopt(shoptFile)
  shoptFile = string(shoptFile)
  py"""
  shoptFile = $shoptFile
  from astropy.io import fits
  f = fits.open(shoptFile)
  polyMatrix = f[0].data
  degree = f[1].data['POLYNOMIAL_DEGREE'][0]
  """
  polynomialMatrix = convert(Array{Float64,3}, py"polyMatrix")
  degree = convert(Int64, py"degree")

  return polynomialMatrix, degree
end

function p(u,v, polMatrix, degree)
  #Read in degree of Polynomial
  #Read in Size of Star Vignet
  r,c = size(polMatrix,1), size(polMatrix,2)
  star = zeros(r,c)
    
  for a in 1:r
    for b in 1:c

      function objective_function(p, x, y, degree)
        value = 0
        counter = 0
          
        for i in 1:(degree + 1)
          for j in 1:(degree + 1)
            if (i - 1) + (j - 1) <= degree
              counter += 1
              value += p[counter] * x^(i - 1) * y^(j - 1) 
            end
          end
        end
        return value
      end
      star[a,b] = objective_function(polMatrix[a,b,:] ,u,v,degree) # -> Filename Polynomial Matrix
    end
  end
  star = star/sum(star)
  return star
end

