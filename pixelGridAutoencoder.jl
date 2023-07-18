# Define the encoder
encoder = Chain(
                Dense(r*c, 128, leakyrelu),
                Dense(128, 64, leakyrelu),
                Dense(64, 32, leakyrelu),
               )

# Define the decoder
decoder = Chain(
                Dense(32, 64, leakyrelu),
                Dense(64, 128, leakyrelu), #leakyrelu #relu
                Dense(128, r*c, tanh),   #tanh #sigmoid
               )
 
# Define the full autoencoder
autoencoder = Chain(encoder, decoder)

#x̂ = autoencoder(x)
loss(x) = mse(autoencoder(x), x)

# Define the optimizer
optimizer = ADAM()

