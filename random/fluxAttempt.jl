using Flux
#using Flux: onehotbatch, throttle, @epochs, mse
using Images
using PyCall
using Plots

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


# Define the encoder
encoder = Chain(
                Dense(r*c, 128, relu),
                Dense(128, 64, relu),
                Dense(64, 32, relu),
               )
# Define the decoder
decoder = Chain(
                Dense(32, 64, relu),
                Dense(64, 128, relu),
                Dense(128, r*c, sigmoid),
               )

# Define the full autoencoder
autoencoder = Chain(encoder, decoder)

# Define the loss function (mean squared error)
loss(autoencoder, x, x̂) = Flux.mse(autoencoder(x), x̂)

# Format some random image data
data = reshape(starCatalog[1], length(starCatalog[1])) 

#opt = ADAM()
opt = Flux.setup(Adam(), autoencoder)

# Train the autoencoder
@time begin
  for epoch in 1:1000
    Flux.train!(loss, autoencoder, [(data, data)], opt) #Flux.params(autoencoder))
  end
end


# Take a sample input image
input_image = reshape(starCatalog[1], length(starCatalog[1]))

# Pass the input image through the autoencoder to get the reconstructed image
reconstructed_image = autoencoder(input_image)

# Reshape the images to 28x28 for display
input_image = reshape(input_image, (r, c))

reconstructed_image = reshape(reconstructed_image, (r, c))

# Convert the images to the appropriate format (e.g., grayscale)

# Save the input and reconstructed images
savefig(plot(heatmap(input_image), 
             heatmap(reconstructed_image),
             layout = (1,2),
             size = (1920,1080)), 
             "flux.png")

