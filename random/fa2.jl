using Plots
using Flux
using Flux: @epochs
using PyCall
function cataloging()
  py"""
  import numpy as np
  from astropy.io import fits
  from astropy.table import Table, hstack, vstack
  f = fits.open('/home/eddieberman/research/mcclearygroup/shopt/datadir/mosaic_nircam_f444w_COSMOS-Web_60mas_v0_1_starcat.fits')
  vignets = f[1].data['VIGNET']
  l = len(vignets)
  """ 

  v = py"vignets"
  catalog = py"list(map(np.array, $v))"
  r = size(catalog[1], 1)
  c = size(catalog[1], 2)
  return catalog, r, c, length(catalog)
end
 
starCatalog, r, c, itr = cataloging()
temp = [reshape(arr, length(arr)) for arr in starCatalog]
temp = hcat(temp...)
trainData = temp 

println("cataloging complete")

# Latent dimensionality, # hidden units.
Dz, Dh = 2, 500

# encoder
g = Chain(Dense(r*c, Dh, tanh), Dense(Dh, Dz))
# decoder
f = Chain(Dense(Dz, Dh, tanh), Dense(Dh, r*c, σ))
# model
sae = Chain(g,f)
# loss
loss(x, x̂) = Flux.mse(sae(x), x̂)
# callback
#evalcb = throttle(() -> ( p=rand(1:N, M); @show(loss(X[:, p],Y[p]))), 20) 
# optimization
opt = ADAM()
# parameters
#ps = params(sae);

@epochs 300 Flux.train!(loss, Flux.params(sae), [(trainData, trainData)], opt)

input_image = reshape(starCatalog[1], length(starCatalog[1]))

#Pass the input image through the autoencoder to get the reconstructed image
reconstructed_image = sae(input_image)
input_image = reshape(input_image, (r, c))
 
reconstructed_image = reshape(reconstructed_image, (r, c))
savefig(plot(heatmap(input_image), heatmap(reconstructed_image), layout = (1,2), size = (1920,1080)), "flux.png")
