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

#x̂ = autoencoder(x)
loss(x) = mse(autoencoder(x), x)
# + msle(shift_positive(x), shift_positive(autoencoder(x)))

# Define the optimizer
optimizer = ADAM()

