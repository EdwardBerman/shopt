---
title: 'ShOpt.jl | A Julia Library for Empirical Point Spread Function Characterization of JWST NIRCam Data'
tags:
  - JWST
  - Deep Learning
  - Julia
  - Point Spread Function

authors:
  - name: Edward Berman
    orcid: 0000-0002-8494-3123
    corresponding: true
    affiliation: 1
  - name: Jacqueline McCleary
    orcid: 0000-0002-9883-7460
    corresponding: true
    affiliation: 1
affiliations:
 - name: Northeastern University, USA
   index: 1
date: 23 August 2023
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
aas-doi: 10.3847/xxxxx
aas-journal: Astronomical Journal
---

# Summary
## Introduction
When astronomers capture images of the night sky, several factors -- ranging from diffraction and optical aberrations to atmospheric turbulence and telescope jitter -- affect the incoming light. The resulting distortions are summarized in the image's point spread function (PSF), a mathematical model that describes the response of an optical system to an idealized point of light. The PSF can obscure or even mimic the astronomical signal of interest, making its accurate characterization essential.  By effectively modeling the PSF, we can predict image distortions at any location and proceed to deconvolve the PSF, ultimately reconstructing distortion-free images.

  The PSF characterization methods used by astronomers fall into two main classes: forward-modeling approaches, which use physical optics propagation based on models of the optics, and empirical approaches, which use stars as fixed points to model and interpolate the PSF across the rest of the image. (Stars are essentially point sources before their light passes through the atmosphere and telescope, so the shape and size of their surface brightness profiles define the PSF at that location.) Empirical PSF characterization proceeds by first cataloging the observed stars, separating the catalog into validation and training samples, and interpolating the training stars across the field of view of the camera. After training, the PSF model can be validated by comparing the reserved stars to the PSF model's prediction.

Shear Optimization with `ShOpt.jl` introduces modern techniques, tailored to James Webb Space Telescope (JWST) imaging, for empirical PSF characterization across the field of view. ShOpt has two modes of operation: approximating stars with analytic profiles, and a more realistic pixel-level representation. Both modes take as input a catalog with image cutouts -- or "vignettes" -- of the stars targeted for analysis.

## Analytic profile mode
A rough idea of the size and shape of the PSF can be obtained by fitting stars with analytic profiles. We adopt a multivariate Gaussian profile, as it is computationally cheap to fit one to an image. That is, Gaussian profiles are easy to differentiate and don't involve any numeric integration or other costly steps to calculate. Fitting other common models, such as a Kolmogorov profile, involves numeric integration and thus take much longer to fit. Moreover, the JWST point spread function is very "spikey" (cf. Figure 1). As a result, analytic profiles are limited in their ability to model the point spread function anyway, making the usual advantages of a more expensive analytic profile moot.   

![The plot on the left shows the average cutout of all stars in a supplied catalog. The plot in the middle shows the average point spread function model for each star. The plot on the right shows the average normalized error between the observed star cutouts and the point spread function model.](spikey.png)

Our multivariate gaussian is parameterized by three variables, $[s, g_1, g_2]$, where $s$ corresponds to size and $g_1 , g_2$ correspond to shear. A shear matrix has the form $$\begin{pmatrix}
1 + g_1 & g_2 \\
g_2 & 1 - g_1
\end{pmatrix}
$$. Given a point $[u, v]$, we obtain coordinates $[u' , v']$ by applying a shear and then a scaling by $\frac{s}{\sqrt{1 - g_1^2 - g_2^2}}$. Then, we choose $f(r) :=  Ae^{-r^2}$ to complete our fit, where $A$ makes the fit sum to unity over the cutout of our star. After we fit this function to our stars with `Optim.jl` [@Mogensen2018] and `ForwardDiff.jl` [@RevelsLubinPapamarkou2016], we interpolate the parameters across the field of view according to position. Essentially, each star is a datapoint, and the three variables are treated as polynomials in the focal plane. We express positions in sky (astrometric) coordinates ($u$, $v$), as opposed to pixel coordinates ($x$,$y$) measured in detector pixels.

### Notation
1. For the set $B_2(r)$, we have:

   $$
   B_2(r) \equiv \{ [x,y] : x^2 + y^2 < 1 \} \subset \mathbb{R}^2
   $$

2. For the set $\mathbb{R}_+$, we have:

   $$
   \mathbb{R}_+ \equiv \{ x : x > 0 \} \subset \mathbb{R}
   $$

3. For the Cartesian product of sets $A$ and $B$, we have:

   $$
   A \times B \equiv \{(a, b): a \in A, b \in B \}
   $$

### Analytic methods
`ShOpt.jl`'s analytic profile fitting takes inspiration from a number of algorithms outside of astronomy, notably SE-Sync [@doi:10.1177/0278364918784361], an algorithm that solves the robotic mapping problem by considering the manifold properties of the data. With sufficiently clean data, the SE-Sync algorithm will descend to a global minimum constrained to the manifold $SE(d)^n / SE(d)$. Following suit, we are able to put a constraint on the solutions we obtain to $[s, g_1, g_2]$ to a manifold. The solution space to $[s, g_1, g_2]$  is constrained to the manifold $B_2(r) \times \mathbb{R}_{+}$ [@Bernstein_2002]. The existence of the constraint on shear is well known; nevertheless, the parameter estimation task is usually framed as an unconstrained problem  [@Jarvis_2020]. For a more rigorous treatment of optimization on manifolds see [@AbsMahSep2008] and [@boumal2023intromanifolds]. `Julia` offers support for working with manifolds through the `Manopt` framework, which we may leverage in future releases [@Bergmann2022].

![LFBGS algorithm used to find parameters subject to the cylindrical constraint. s is arbitrarily capped at 1 as a data cleaning method.](pathToPoint.png)

## Pixel grid mode
A more complete description of the PSF can be obtained using the image pixels themselves as a basis, with an interpolation function to model PSF variation across the field of view. `ShOpt.jl` provides two modes for these pixel grid fits: `PCA mode` and `Autoencoder mode`. `PCA mode`, outlined below, reconstructs its images using the first $n$ principal components. `Autoencoder mode` uses a neural network to reconstruct the image from a lower dimensional latent space. The network code written with `Flux.jl` is also outlined below [@innes:2018]. Both modes provide the end user with tunable parameters that allow for both perfect reconstruction of star cutouts and low dimensional representations. The advantage of these modes is that they provide good reconstructions of the distorted images that can learn the key features of the point spread function without overfitting the background noise. In this way it generates a datapoint for our algorithm to train on and denoises the image in one step. In both cases, the input star data is cleaned by first fitting an analytic (Gaussian) PSF profile and rejecting size outliers. As in the analytic profile case, star positions are expressed directly in sky (astrometric) coordinates ($u$, $v$) rather than pixel coordinates ($x$,$y$). In the pixel grid modes, we also model each pixel in the star stamp as polynomial to be interpolated across the field of view. That is, each pixel in position $(i,j)$ of a star cutout gets its own polynomial, interpolated over $k$ different star cutouts at different locations in the focal plane.

### Pixel grid methods

`PCA mode`
```Julia
function pca_image(image, ncomponents)    
  #Load img Matrix
  img_matrix = image

  # Perform PCA    
  M = fit(PCA, img_matrix; maxoutdim=ncomponents)    

  # Transform the image into the PCA space    
  transformed = MultivariateStats.transform(M, img_matrix)    

  # Reconstruct the image    
  reconstructed = reconstruct(M, transformed)    

  # Reshape the image back to its original shape    
  reconstructed_image = reshape(reconstructed, size(img_matrix)...)    
end    
```
`Autoencoder mode`
```Julia
# Encoder    
encoder = Chain(    
                Dense(r*c, 128, leakyrelu),    
                Dense(128, 64, leakyrelu),    
                Dense(64, 32, leakyrelu),    
               )    
#Decoder
decoder = Chain(    
                Dense(32, 64, leakyrelu),    
                Dense(64, 128, leakyrelu),    
                Dense(128, r*c, tanh),    
               )    
#Full autoencoder
autoencoder = Chain(encoder, decoder)    

#x_hat = autoencoder(x)    
loss(x) = mse(autoencoder(x), x)    

# Define the optimizer    
optimizer = ADAM()    

```

# Statement of need
Empirical PSF characterization tools like PSFEx [@2011ASPC] and PIFF [@Jarvis_2020] are widely popular in astrophysics. However, the quality of PIFF and PSFEx models tends to be quite sensitive to the parameter values used to run the software, with optimization sometimes relying on brute-force guess-and-check runs. PIFF is also notably inefficient for large, well-sampled images, taking hours in the worst cases. The James Webb Space Telescope's (JWST) Near Infrared Camera (NIRCam) offers vast scientific opportunities (e.g., [@casey2023cosmosweb]); at the same time, this unprecendented data brings new challenges for PSF modeling:

(1) Analytic functions like Gaussians are incomplete descriptions of the NIRCam PSF, as evident from Figure 1.  This calls for well-thought-out, non-parametric modeling and diagnostic tools that can capture the full dynamic range of the NIRCam PSF. `ShOpt` provides these models and diagnostics out of the box.

(2) The NIRCam detectors have pixel scales of 0.03 (short wavelength channel) and 0.06 (long wavelength channel) arcseconds per pixel [@10.1117/12.489103; @BSPIE; @20052005SPIE]. At these pixel scales, star vignettes need to be at least $131$ by $131$ pixels across to fully capture the wings of the PSFs (4-5 arcseconds). These vignette sizes are 3-5 times larger than the ones used in surveys such as DES [@Jarvis_2020] and SuperBIT [@mccleary2023lensing] and force us to evaluate how well existing PSF fitters scale to this size. `ShOpt` has been designed for computational efficiency and aims to meet the requirements of detectors like NIRCam.  

# State of the Field
There are several existing empirical PSF fitters, in addition to a forward model of the JWST PSFs developed by STScI [@Jarvis_2020 ; @2011ASPC; @2014SPIE ; @2012SPIE]. We describe them here and draw attention to their strengths and weaknesses to motivate the development of `ShOpt.jl`. As described in the statement of need, `PSFex` was one of the first precise and general purpose tools used for empirical PSF fitting. However, the Dark Energy Survey collaboration reported small but noticeable discrepancies between the sizes of `PSFex` models and the sizes of observed stars. They also reported ripple-like patterns in the spatial variation of star-PSF residuals across the field of view [@Jarvis_2020], which they attributed to the astrometric distortion solutions for the Dark Energy Camera.

These findings motivated the Dark Energy Survey's development of `PIFF` (Point Spread Functions in the Full Field of View). `PIFF` works in sky coordinates on the focal plane, as opposed to image pixel coordinates used in `PSFex`, which minimized the ripple patterns in the star-PSF residuals and the PSF model size bias. (Based on the DES findings, `ShOpt` also works directly in sky coordinates.) `PIFF` is written in Python, a language with a large infrastructure for astronomical data analysis, for example Astropy [@2022ApJ] and Galsim [@rowe2015galsim]. The choice of language makes `PIFF` software more accessible to programmers in the astrophysics community than `PSFex`, which was first written in `C` twenty-five years ago and much less approachable for a community of open source developers. As an aside, one of the motivations of `ShOpt` was to write astrophysics specific software in `Julia`, because `Julia` provides a nice balance of readability and speed with its high-level functional paradigm and just-in-time compiler.  

While WebbPSF provides highly precise forward models of the JWST PSF, these models are defined for single-epoch exposures [@2014SPIE; @2012SPIE]. Much of the NIRCam science is accomplished with image mosaics -- essentially, the combination of single exposure detector images into a larger, deeper image. The rotation of the camera between exposures, the astrometric transformations and resampling of images before their combination into a mosaic, and the mosaic's large area all make the application of WebbPSF models to mosaics a non-trivial procedure. Additionally, some recent work being done to generate hybrid PSF models, which add an empirical correction to forward-model PSFs, for single-epoch exposures [@lin2023hybpsf]. At the time of writing, there is no widely available software to do this.  

# Future Work

The COMOS-Web survey is the largest JWST extragalactic survey according to area and prime time allocation [@casey2023cosmosweb], and takes up $0.54 ~deg^2$ [@10.1117/12.925447; @Rieke_2023]. This is a large enough portion of the sky that we should prepare to see significant PSF variation across the field of view because of astrometric distortions. Thus, COSMOS-Web data provides `ShOpt` with an oppurtunity to validate PIFF's correction for handling PSF variations and test how impactful (or not impactful) PSFex's size bias is.

We speculate that petal diagrams may be able to approximate the spikey natures of JWST PSFS. Consider $r = A \cos(k\theta + \gamma)$, shown below in figure 3 for different $[A, k]$ values where $\gamma = 0$. In practice, $[A, k, \gamma]$ could be learnable parameters. Moreover, we could do this for a series of trigonmetric functions to get petals of different sizes. We could then choose some $f(r) \propto \frac{1}{r}$ such that the gray fades from black to white. We would define $f(r)$ piece wise such that it is $0$ outside of the petal and decreases radially with $r$ inside the petal.

![Petal Diagram](petals.png)

# Acknowledgements
This material is based upon work supported by a Northeastern University Undergraduate Research and Fellowships PEAK Experiences Award. E.B. was also supported by a Northeastern University Physics Department Co-op Research Fellowship. Support for COSMOS-Web was provided by NASA through grant JWST-GO-01727 and HST-AR-15802 awarded by the Space Telescope Science Institute, which is operated by the Association of Universities for Research in Astronomy, Inc., under NASA contract NAS 5-26555. This work was made possible by utilizing the CANDIDE cluster at the Institut d’Astrophysique de Paris. Further support was provided by Research Computers at Northeastern University. Additionally, E.B. thanks Professor David Rosen for giving some valuable insights during the early stages of this work. The authors gratefully acknowledge the use of simulated and real data from the COSMOS-Web survey in developing ShOpt, as well as many conversations with COSMOS-Web scientists. 

# References
