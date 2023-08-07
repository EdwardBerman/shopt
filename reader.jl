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
  s_matrix = f[6].data['s_MATRIX']
  g1_matrix = f[6].data['g1_MATRIX']
  g2_matrix = f[6].data['g2_MATRIX']
  """
  polynomialMatrix = convert(Array{Float64,3}, py"polyMatrix")
  degree = convert(Int64, py"degree")
  s_matrix = convert(Array{Float64,1}, py"s_matrix")
  g1_matrix = convert(Array{Float64,1}, py"g1_matrix")
  g2_matrix = convert(Array{Float64,1}, py"g2_matrix")

  return polynomialMatrix, degree, s_matrix, g1_matrix, g2_matrix
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

function analytic_profile(u,v, s_matrix, g1_matrix, g2_matrix, radial_function)
  s = s_matrix[1]*u^3 + s_matrix[2]*v^3 + s_matrix[3]*u^2*v + s_matrix[4]*v^2*u + s_matrix[5]*u^2 + s_matrix[6]*v^2 + s_matrix[7]*u*v + s_matrix[8]*u + s_matrix[9]*v + s_matrix[10]
  g1 = g1_matrix[1]*u^3 + g1_matrix[2]*v^3 + g1_matrix[3]*u^2*v + g1_matrix[4]*v^2*u + g1_matrix[5]*u^2 + g1_matrix[6]*v^2 + g1_matrix[7]*u*v + g1_matrix[8]*u + g1_matrix[9]*v + g1_matrix[10]
  g2 = g2_matrix[1]*u^3 + g2_matrix[2]*v^3 + g2_matrix[3]*u^2*v + g2_matrix[4]*v^2*u + g2_matrix[5]*u^2 + g2_matrix[6]*v^2 + g2_matrix[7]*u*v + g2_matrix[8]*u + g2_matrix[9]*v + g2_matrix[10]
  return radial_function(s,g1,g2)
end
