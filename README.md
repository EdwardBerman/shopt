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


Known Issues: Need to take stamps of images to focus on the actual PSF, Some plots have issues with colorbars 

Tourists Guide to this program

shopt.jl 
> A runner script for all functions in this software

dataPreprocessing.jl
> A wrapper for python code to handle fits files and dedicated file to deal with data cleaning and adding noise to test robustness of the software

plot.jl 
> A dedicated file to handle all plotting
