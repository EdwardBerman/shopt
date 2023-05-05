using PyCall

# Create a 2D NumPy array in Python
py"""
import numpy as np
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
"""

# Convert the NumPy array to a Julia array
julia_array = convert(Array{Float64,2}, py"a")

# Print the Julia array
println(julia_array)

