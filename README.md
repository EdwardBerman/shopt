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

Tourists Guide to this program:

shopt.jl 
> A runner script for all functions in this software

dataPreprocessing.jl
> A wrapper for python code to handle fits files and dedicated file to deal with data cleaning and adding noise to test robustness of the software

plot.jl 
> A dedicated file to handle all plotting

radialProfiles.jl 
> Contains analytic profiles such as a Gaussian Fit and a kolmogorov fit

analyticCGD.jl 
> Provides the necessary arguments (cost function and gradient) to the optimize function for analytic fits 

pixelGridCGD.jl 
> Provides the necessary arguments (cost and gradient) to do a pixel grid Optimization

fluxNormalizer.jl 
> A function to determine A such that analytic profiles sum to unity

ellipticityNormalizer.jl 
> A function that maps the norm of a vector in Euclidean Space inside of an open ball, scales components appropriately such that g1 and g2 are in an open ball in 2d space

interpolate.jl 
> For Point Spread Functions that vary across the Field of View, interpolate.jl will fit a 3rd degree polynomial in u and v to show how each of the pixel grid parameters change across the ra and dec

outliers.jl 
> Contains two functions for identifying and removing outliers from a list

LICENSE
> MIT LICENSE

README.md
> User guide, Dependencies, etc.
