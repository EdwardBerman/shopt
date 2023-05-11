# shopt
Shear Optimization with Shopt.jl, a julia library for empirical point spread function characterizations

To run `shopt.jl`
Run ```julia shopt.jl [eventually the fits file you want to run] [configdir] [outdir] [datadir]```

Dependencies:
| Julia            | Python   |
|------------------|----------|
| Plots            | treecorr |  
| ForwardDiff      | astropy  |  
| LinearAlgebra    | webbpsf  |  
| Random           |          |  
| Distributions    |          |  
| SpecialFunctions |          |  
| Optim            |          |  
| IterativeSolvers |          |  
| QuadGK           |          |  
| PyCall           |          |   

Julia: Plots, ForwardDiff, LinearAlgebra, Random, Distributions, SpecialFunctions, Optim, IterativeSolvers, QuadGK, PyCall
  Python: Astropy, treecorr

Known Issues: Need to take stamps of images to focus on the actual PSF, Some plots have issues with colorbars 
