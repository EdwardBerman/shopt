---
title: 'ShOpt.jl | A Julia Library for Empirical Point Spread Function Characterization of JWST NIRCam Images'
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
    affiliation: 1
affiliations:
 - name: Northeastern University, USA
   index: 1
date: 23 August 2023
bibliography: paper.bib

---

# Summary
When cosmologists try to take pictures of space, a combination of the photometry of the camera and atmospheric affects distort the light that comes from stars. Stars are examples of what astronomers call point sources, and so the aptly named point spread function (PSF) is a mathematical model that quantifies exactly how the light is being distorted. The point spread function takes as input a delta function and a position and outputs a lensed image. The goal of empirical point spread function characterization is to be able to point to any position on your camera and predict what the lensed star looks like. Once we have a model that can do this well, we can deconvolve our images with the point spread function to obtain what the image would look like in the absense of lensing. The empirical way to do this is to take our images of lensed stars and seperate them into training and validation set. Our point spread function will be found by interpolating the training stars across the field of view of the camera and validated by comparing the reserved stars to the point spread function's prediction.

Shear Optimization with `ShOpt.jl` introduces modern techniques for empirical point spread function characterization across the full field of view tailored to the data captured by the James Webb Space Telescope. To first order, we can approximate our images with analytic profiles. We adopt a multivariate gaussian because it is computationally cheap to fit to an image. This function is parameterized by three variables, $[s, g_1, g_2]$, where $s$ corresponds to size and $g_1 , g_2$ correspond to shear. After we fit this function to our stars with `Optim.jl` and `ForwardDiff.jl` [@Mogensen2018; @RevelsLubinPapamarkou2016], we interpolate the parameters across the field of view according to position. Essentially, each star is a datapoint, and the three variables are given polynomials in focal plane coordinates of degree $n$, where $n$ is supplied by the user. For a more precise model, we also give each pixel in our images a polynomial and interpolate it across the field of view. This is referred to in the literature as a pixel grid fit [@Jarvis_2020]. 

`ShOpt.jl` takes inspiration from a number of algorithms outside of astronomy. Mainly, SE-Sync [@doi:10.1177/0278364918784361], an algorithm that provides a certifiably correct solution to a robotting mapping problem by considering the manifold properites of the data. We borrow this idea to put a constraint on the solutions we obtain to $[s, g_1, g_2]$. [@Bernstein_2002] outlined the manifold properties of shears for us, so we knew from the get go that our solution was constrained to the manifold $B_2(r) \times \mathbb{R}_{+}$. While it was known that this constrain existed in the literature, the parameter estimation tasked had been framed as an unconstrained problem prior to our work  [@Jarvis_2020]. For a more rigorous treatment of optimization on manifolds see [@AbsMahSep2008] and [@boumal2023intromanifolds]. `Julia` has lots of support for working with manifolds with `Manopt`, which we may leverage in future releases [@Bergmann2022]. 

`ShOpt.jl` provides two modes for pixel grid fits, `PCA mode` and `Autoencoder mode`. Each mode provides the end user with tunable parameters that allow for both perfect reconstruction of the model vignets and low dimensional representations. The advantage of these modes is that they provide good reconstructions of the lensed images while fixating on the actual star and not the background noise. In this way it generates a datapoint for our empirical point spread function to learn and denoises the image in one step.

`PCA mode`, outlined here, reconstructs it's images using the first n principal components.
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
`Autoencoder mode` uses a neural network to reconstruct the image from a lower dimensional latent space. The network code written with `Flux.jl` is below [@innes:2018]
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
While there are many existing empirical PSF fitters, they were created as apart of the efforts of other collaborations with their own cameras and science goals. Mainly, The Dark Energy Survey and DESCam [@Jarvis_2020;@2015AJ]. The recent data from the James Webb Space Telescope poses new challenges. 

(1) The James Webb PSFs are not well approximated by analytic profiles. This calls for well thought out parametric free models that can capture the full dynamic range of the Point Spread Function without fixating on the noise in the background.  

(2) The NIRCam detectors measure 0.03"/pix [@Gardner_2006]. To capture an accurate description of the point spread function at this scale we need images that are $131$ by $131$ to $261$ by $261$ pixels across. These vignet sizes are much larger in comparison to the sizes needed for previous large scale surveys such as DES [@Jarvis_2020] and SuperBIT [@mccleary2023lensing] and forces us to evaluate how well existing PSF fitters scale to this size.


# State of the Field
There are several existing empirical PSF fitters in addition to a theoretical prediction of the James Webb PSFs developed by STScI [@Jarvis_2020 ; @2011ASPC; @2014SPIE ; @2012SPIE]. We describe them here and draw attention to their strenghts and weaknesses to motivate the development of `ShOpt.jl`. The first empirical PSF fitter developed was `PSFex`. It used statistical methods that were natural starting points for the problem at hand and prove to be sufficient in many cases to this day. However, as Mike Jarvis and his collaborators with DES noticed, `PSFex` produced a systematic size bias of the point spread function with how it calculated spatial variation across the field of view [@Jarvis_2020] 

`PIFF` (Point Spread Functions in the Full Field of View) followed `PSFex` in the effort to correct this issue. The DES camera was $2.2$ degrees across, which was large enough for the size bias to become noticable for their efforts. `PIFF` works in focal plane coordinates as opposed to sky coordinates which fixes the systematic size bias. Jarvis and DES also used the `Python` libraries of astropy [@2022ApJ] and Galsim [@rowe2015galsim] to make the software more accessible than PSFex to programmers in the astrophysics community. PSFex was written in `C` and had been active for more than 20 years before the systematic size bias was discovered. Due to being so old and written in a low level language it is much less approachable for a community of open source developers. One of the motivations of `ShOpt` was to write astrophysics specific software in `Julia`, because `Julia` provides a nice balance of readability and speed with it's high level functional paradigm and just in time compiler.

While we do have theoretical models of the James Webb PSF, there is yet to be any validation of these models on real data in the literature. Additionally, these models are for single exposure images. The James Webb images have both single exposure and mosaiced images [@2014SPIE; @2012SPIE]. Mosaiced images are essentially single exposure detector images concatenated together side by side. The PSF models for single exposures do not generalize to the mosaics, so empirical models are all we have for those images.  

The COMOS-Web survey is the largest extragalactic survey according to area and prime time allocation [@casey2023cosmosweb], and takes up $0.54 deg^2$ [@10.1117/12.925447; @Rieke_2023]. This is a large enough portion of the sky that we should prepare to see a lot of variation across the field of view. This gives `ShOpt` the oppurtunity to validate PIFF's correction for handeling PSF variations and underscore just how impactful (or not impactful) PSFex's size bias is. 

# Acknowledgements
This project was made possible by The Northeastern Physics Department and Northeastern Undergraduate Research and Fellowships via the Physics Research Co-Op Fellowship and PEAK Ascent Award respectively. Support for this work was provided by NASA through grant JWST-GO-01727 and HST-AR- 15802 awarded by the Space Telescope Science Institute, which is operated by the Association of Universities for Research in Astronomy, Inc., under NASA contract NAS 5-26555. This work was made possible by utilizing the CANDIDE cluster at the Institut d’Astrophysique de Paris. Finally, we would like to thank Northeastern Research Computing for access to their servers.

# References
