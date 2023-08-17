## Table of Contents
- [About](#about)
  - [Analytic Profile Fits](#analytic-profile-fits)
  - [Pixel Grid Fits](#pixel-grid-fits)
  - [Interpolation Across the Field of View](#interpolation-across-the-field-of-view)
- [Inputs and Outputs](#inputs-and-outputs)
  - [Inputs](#inputs)
  - [Outputs](#outputs)
- [Running](#running)
  - [Command](#command)
  - [Dependencies](#dependencies)
  - [Set Up](#set-up)
- [Program Architecture](#program-architecture)
- [Config / YAML Information](#config--yaml-information)
- [Known Issues](#known-issues)
- [Contributors](#contributors)
- [Further Acknowledgements](#further-acknowledgements)


## About
 [![License](https://img.shields.io/pypi/l/jax-cosmo)](https://github.com/EdwardBerman/shopt/blob/main/LICENSE) [![All Contributors](https://img.shields.io/badge/all_contributors-2-orange.svg?style=flat-square)](#contributors)

**Shear Optimization** with **ShOpt.jl**, a julia library for empirical point spread function characterizations. We aim to improve upon the current state of Point Spread Function Modeling by using Julia to leverage performance gains, use a different mathematical formulation than the literature to provide more robust analytic and pixel grid fits, improve the diagnostic plots, and add features such as wavelets and shapelets. At this projects conclusion we will compare to existing software such as PIFF and PSFex. Work done under [McCleary's Group](https://github.com/mcclearyj).

Start by **Cloning This Repository**. Then see **TutorialNotebook.ipynb** or follow along the rest of this **README.md** to get started!

### Analytic Profile Fits 
We adopt the following procedure to ensure our gradient steps never take us outside of our constraints

![image](READMEassets/reparameterization.png)

### Pixel Grid Fits                                                                                                        

#### PCA Mode 
We used the first n weights of a Principal Component Analysis and use that to construct our PSF in addition to a smoothing kernel to account for aliasing 

#### Autoencoder Mode
For doing Pixel Grid Fits we use an autoencoder model to reconstruct the Star                                              
![image](READMEassets/nn.png) 

### Interpolation Across the Field of View
[s, g1, g2] are all interpolated across the field of view. Each Pixel is also given an interpolation across the field of view for an nth degree polynomial in (u,v), where n is supplied by the user

## Inputs and Outputs
Currently, the inputs are JWST Point Spread Functions source catalogs. The current outputs are images of these Point Spread Functions, Learned Analytic Fits, Learned Pixel Grid Fits, Residual Maps, Loss versus iteration charts, and p-value statisitcs. Not all functionality is working in its current state. Planned functionality for more Shear checkplots.

### Inputs 

| Image                             | Description                        |
|-----------------------------------|------------------------------------|
| ![image](READMEassets/input.png)  | Star Taken From Input Catalog      |
| shopt.yml                         | Config File for Tunable Parameters | 
| \* starcat.fits                   | Star Catalog to take  vignets from |

### Outputs

| Image                                              | Description                                                                                                                         |
|----------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------|
| summary.shopt                                      | Fits File containing summary statistics and information to reconstruct the PSF                                                      |
| ![image](READMEassets/pgf9.png)                    | Pixel Grid Fit for the Star Above                                                                                                   |
| ![image](READMEassets/hmresid.png)                 | Residual Map for Above Model and Fit                                                                                                |
| ![image](READMEassets/s_uv.png)                    | s varying across the field of view                                                                                                  |
| ![image](READMEassets/g1_uv.png)                   | g1 varying across the field of view                                                                                                 |
| ![image](READMEassets/g2_uv.png)                   | g2 varying across the field of view                                                                                                 |
| ![image](READMEassets/parametersHistogram.png)     | Histogram for learned profiles for each star in an analytic fit with their residuals                                                |
| ![image](READMEassets/parametersScatterplot.png)   | Same data recorded as a scatterplot with and without outliers removed and with error bars                                           |

NB: This is not a comprehensive list, only a few cechkplots are presented. See the shopt.yml to configure which plots you want to see and save!

## Running
### Command 
To run `shopt.jl`

First use Source Extractor to create a catalog for ShOpt to accept and save this catalog in the appropriate directory

Run ```julia shopt.jl [configdir] [outdir] [catalog]```

There is also a shell script that runs this command so that the user may call shopt from a larger program they are running

### Dependencies 
Not all of these will be strictly necessary depending on the checkplots you produce, but for full functionality of ShOpt the following are necessary. Source Extractor (or Source Extractor ++) is also not a strict dependency, but in practice one will inevitably install to generate a catalog.

| Julia            | Python     | Binaries | Julia          | Julia          |
|------------------|------------|----------|----------------|----------------|
| Plots            | matplotlib | SEx      | ProgressBars   | BenchmarkTools |
| ForwardDiff      | astropy    |          | UnicodePlots   | Measures       |
| LinearAlgebra    | numpy      |          | CSV            | Dates          |
| Random           |            |          | FFTW           | YAML           |
| Distributions    |            |          | Images         | CairoMakie     |
| SpecialFunctions |            |          | ImageFiltering | Flux           |
| Optim            |            |          | DataFrames     | QuadGK         |
| IterativeSolvers |            |          | PyCall         | Statistics     |

### Set Up 
Start by cloning this repository. There are future plans to release ShOpt onto a julia package repository, but for now the user needs these files contents.

The dependencies can be installed in the Julia REPL. For example:
```julia
import Pkg; Pkg.add("PyCall")
```

We also provide dependencies.jl, which you can run to download all of the Julia libraries automatically by reading in the imports.txt file. Simply run ` julia dependencies.jl ` in the command line. For the three python requirements, you can similarly run `python dependenciesPy.py`. 

For some functionality we need to use wrappers for Python code, such as reading in fits files or converting (x,y) -> (u,v). Thus, we need to use certain Python libraries. Thankfully, the setup for this is still pretty straightfoward. We use PyCall to run these snippets. There are a number of ways to get Julia and Python to interopt nicely. If the Python snippets throw an error, run the following in the Julia REPL for each Python library:

```julia
using PyCall
ENV["PYTHON"] = "/path_desired_python_directory/python_executable"; import Pkg; Pkg.build("PyCall")
pyimport("astropy")
```

If you have a Conda Enviornment setup, you may find it easier to run 
```julia
using PyCall
pyimport_conda("astropy", "ap") #ap is my choice of name and astropy is what I am importing from my conda Enviornment
```

Or Similarly,

```julia
using Conda
Conda.add("astropy", :my_env) #my conda enviornment is named my_env
Conda.add("astropy", "/path/to/directory")
Conda.add("astropy", "/path/to/directory"; channel="anaconda")
```

On the off chance that none of these works, a final method may look like the following 
```julia
using PyCall
run(`$(PyCall.python) -m pip install --upgrade cython`)
run(`$(PyCall.python) -m pip install astropy`) 
```

After the file contents are downloaded the user can run ```julia shopt.jl [configdir] [outdir] [catalog]``` as stated above. Alternatively, they can run the shellscript that calls shopt in whatever program they are working with to create their catalog. For example, in a julia program you may use ```run(`./runshopt.sh [configdir] [outdir] [catalog]`)```

## Program Architecture

TutorialNotebook.ipynb
> Run ShOpt inside of a Jupyter Notebook and learn both how to run the program and how to reconstruct the PSF 

shopt.jl 
> A runner script for all functions in this software

dataPreprocessing.jl
> A wrapper for python code to handle fits files and dedicated file to deal with data cleaning 

dataOutprocessing.jl
> Convert data into a summary.shopt file. Access this data with reader.jl. Produces some additional python plots.

reader.jl
> Get Point Spread Functions at an arbitrary (u,v) by reading in a summary.shopt file 

plot.jl 
> A dedicated file to handle all plotting

radialProfiles.jl 
> Contains analytic profiles such as a Gaussian Fit and a kolmogorov fit

analyticLBFGS.jl 
> Provides the necessary arguments (cost function and gradient) to the optimize function for analytic fits 

pixelGridAutoencoder.jl
> Houses the function defining the autoencoder and other machine learning functions supplied to Flux's training call

interpolate.jl 
> For Point Spread Functions that vary across the Field of View, interpolate.jl will fit a nth degree polynomial in u and v to show how each of the pixel grid parameters change across the (u,v) plane

outliers.jl 
> Contains functions for identifying and removing outliers from a list

powerSpectrum.jl
> Computes the power spectra for a circle of radius k, called iteratively to plot P(k) / k

kaisserSquires.jl
> Computes the Kaisser-Squires array to be plotted

runshopt.sh
> A shell script for running Shopt. Available so that users can run a terminal command in whatever program they are writing to run shopt. 

LICENSE
> MIT LICENSE

README.md
> User guide, Dependencies, etc.

index.md
> For official website

_config.yml
> Also for official website

imports.txt
> List of Julia Libraries used in the program

packages.txt
> List of Python Libraries used in the program

dependencies.jl 
> Download all of the imports from imports.txt automatically

dependenciesPy.py 
> Download all of the imports from packages.txt automatically

## Config / YAML Information
saveYaml
- Set `true` if you want to save the YAML to the output directory for future reference, set to `false` otherwise

NNparams
- epochs
  - Set the Maximum Number of training epochs should the model never reach the minimum gradient of the loss function. Set to `1000` by default
- minGradientPixel
  - A stopping gradient of the loss function for a pixel grid fit. Set to `1e-6` by default

AnalyticFitParams
- minGradientAnalyticModel
  - A stopping gradient of the loss function for an analytic profile fit for input star vignets from a catalog. Set to `1e-6` by default
- minGradientAnalyticLearned
  - A stopping gradient of the loss function for an analytic profile fit     for stars learned by a pixel grid fit. Set to `1e-6` by default
- analyticFitStampSize
  - The box size for the subset of your stamp (see stamp size) you wish to use for analytic profile fitting. Ensure to specify this to be smaller than the stamp size of the vignets themselves. Set to `64` by default, therefore fitting an analytic profile to the middle `64 x 64` pixels of the stamp. 

dataProcessing
- SnRPercentile 
  - Supply a float that represents the percentile below which stars will be filtered by on the basis of signal to noise ratio. Set to `0.33` by default
- sUpperBound
  - Stars fit with an analytic profile are filtered out if their `s` exceeds this upper bound. Set to `1` by default
- sLowerBound
  - Stars fit with an analytic profile are filtered out if their `s` falls beneath this lower bound. Set to `0.075` by default
 
plots
- Set true to plot and save a given figure, false otherwise 

polynomialDegree
- The degree of the polynomial used to interpolate each pixel in the stamp across the field of view. Set to `3` by default

stampSize
- The size of the vignet for which use wish to fit. Used interpolation for oversampling and a simple crop for undersampling. Set to `131` by default to fit and interpolate `131 x 131` pixels

training_ratio
- Before doing a polynomial interpolation, the remaining stars will be divided into training and validation stars based off of this float. Set to `0.8` by default, indicating 80% training stars 20% validation stars

CommentsOnRun
- This is Where You Can Leave Comments or Notes To Self on the Run! Could be very useful if you save the yaml file with each run

## Known Issues
+ kolmogorov radial profile not yet functional

## Contributors
+ Edward Berman
+ Jacqueline McCleary

## Further Acknowledgements                                                                                                             
+ The Northeastern Cosmology Group for Their Continued Support and Guidance                                                           
+ The Northeastern Physics Department and Northeastern Undergraduate Research and Fellowships, for making this project possible with funding from the Northeastern Physics Research Co-Op Fellowship and PEAK Ascent Award respectively   
+ [David Rosen](https://github.com/david-m-rosen), who gave valuable input in the early stages of this project and during his course Math 7223, Riemannian Optimization
+ The COSMOS Web Collaboration for providing data from the James Webb Space Telescope and internal feedback
