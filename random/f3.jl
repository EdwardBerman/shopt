using Flux
using Plots
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

# Define the dimensions of the input data
input_dim = r*c
hidden_dim = 64
latent_dim = 32

# Define the encoder architecture
encoder = Chain(
    Dense(input_dim, hidden_dim, relu),
    Dense(hidden_dim, latent_dim)
)

# Define the decoder architecture
decoder = Chain(
    Dense(latent_dim, hidden_dim, relu),
    Dense(hidden_dim, input_dim)
)

# Define the full autoencoder by chaining the encoder and decoder
autoencoder = Chain(encoder, decoder)

# Define the loss function (mean squared error)
loss(x) = Flux.mse(autoencoder(x), x)

# Generate random 2D matrices as the training data
#train_data = [rand(input_dim) for _ in 1:1000]
temp = [reshape(arr, r*c) for arr in starCatalog]
train_data = temp
# Train the autoencoder
println("Done Cataloging. Training...")
Flux.@epochs 10 Flux.train!(loss, Flux.params(autoencoder), [(x,) for x in train_data], ADAM(0.001))

