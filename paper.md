---
title: 'ShOpt.jl | A Julia Library for Empirical Point Spread Function Characterization of JWST Sources'
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
date: 08 August 2023
bibliography: paper.bib

---

# Summary
Shear Optimization with \texttt{ShOpt.jl} introduces modern techniques for empirical Point Spread Function (PSF) characterization across the Full Field of View tailored to the data captured by the James Webb Space Telescope. 

Shear Optimization with \texttt{ShOpt.jl} strives to advance the mathematical formulation of empirical Point Spread Function Modeling forward while remaining feasible to compute. \texttt{ShOpt.jl} first takes inspiration from robotics algorithms that run on manifold valued data, such as SE-Sync [@doi:10.1177/0278364918784361]. [@Bernstein_2002] outlined the manifold properties of shears which we expand on to provide more robust analytic profile fits. For a more rigorous treament of Optimization Methods on manifold valued data, we encourage you to see [@AbsMahSep2008] and [@boumal2023intromanifolds]. \texttt{ShOpt.jl} also advances the state of fitting a model pixel by pixel of point sources. \texttt{ShOpt.jl} uses two modes for pixel grid fits, \texttt{PCA mode} and \texttt{Autoencoder mode}. The modes are similar in that they provide the end user with tunable parameters that allow for both perfect reconstruction of the model vignets and low dimensional representations. \texttt{PCA mode} approximates the original image by summing the first \texttt{n} principal components, where \texttt{n} is supplied by the user. We also introduce \texttt{Autoencoder mode}, which uses the deep learning autoencoder architecture to learn what the model point spread function should look like and is robust enough to provide a good fit even in the presence of high signal to noise due to the nonlinearity of the network. 

The programming language and paradigm are an integral part of the software. \texttt{ShOpt.jl} is written in \texttt{Julia}, which enables us to write code that can run quickly while also being readable enough to be accessible for an open source community of \texttt{Julia} programmers to write in their own functions as they see fit. \texttt{Julia} also an abundance of support for working with manifolds such as \texttt{Manopt}, which may be pertinent in future releases for advances in fitting analytic profiles [@Bergmann2022].

```Julia
function pca_image(image_path, ncomponents)    
  # Load the image    
  img_matrix = image_path    
    
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
While there are many existing empirical PSF fitters, they were created as apart of the efforts of The Dark Energy Survey \textbf{[cite]}. The recent data from the James Webb Space Telescope poses new challenges. 

(1) The James Webb PSFs are not well approximated by analytic profiles. This calls for well thought out parametric free models that can capture the full dynamic range of the Point Spread Function without fixating on the noise in the background.  

(2) The NIRCam detectors measure 0.03"/pix (cite). To capture an accurate description of the Point Spread Function at this scale we need images that are $131$ by $131$ to $261$ by $261$ pixels across. These vignet sizes are much larger in comparison to the sizes needed for previous large scale surveys such as DES [@Jarvis_2020] and SuperBIT [@mccleary2023lensing] and forces us to evaluate how well existing PSF fitters scale to this size.


# State of the Field
There are several existing empirical PSF fitters in addition to a theoretical prediction of the James Webb PSFs developed by STScI. We describe them here and draw attention to their strenghts and weaknesses to motivate the development of \texttt{ShOpt.jl}. The first empirical PSF fitter developed was \texttt{PSFex}. It used statistical methods that were natural starting points for the problem at hand and prove to be sufficient in many cases to this day. Moreover, \texttt{PSFex} was written in \texttt{C}, a compiled language known for it's speed of computation, including for tasks such as numerical linear algebra and optimization problems. However, as Mike Jarvis and his collaborators with DES noticed, \texttt{PSFex} produced a systematic size bias of the Point Spread Function with how it calculated spatial variation across the field of view. \textbf{[cite]}

\texttt{PIFF} (Point Spread Functions in the Full Field of View) followed \texttt{PSFex} in the effort to correct this issue. The DES camera was $2.2$" across, which was large enough for the size bias to become noticable for their efforts. \texttt{PIFF} works in focal plane coordinates as opposed to sky coordinates which fixes the systematic size bias. \texttt{PIFF} was written in \texttt{Python}, a language with much more infrastructure built to do computations relevant to Astronomy. The cost of working with such robust tools is that \texttt{PIFF} runs much slower than \texttt{PSFex}. 

Finally, we do have theoretical models of the PSF. The issue is that these models are for single exposure images. The James Webb images have both single exposure and mosaiced images, and the theoretical models are not the same.

# Citations
[@author2023title]

# Acknowledgements
This project was made possible by The Northeastern Physics Department and Northeastern Undergraduate Research and Fellowships via the Physics Research Co-Op Fellowship and PEAK Ascent Award respectively.

# References
