

using Flux
using Flux: onehotbatch, throttle, @epochs, mse
using BSON
using Base.Iterators: repeated, partition
using FluxJS
using MLDatasets # FashionMNIST
using ColorTypes: N0f8, Gray
using Images
     
const Img = Matrix{Gray{N0f8}}

function prepare_train()
    # load full training set
    train_x, train_y = FashionMNIST.traindata() # 60_000

    trainrange = 1:60_000 # 1:60_000
    imgs = Img.([train_x[:,:,i] for i in trainrange])
    
    # Stack into 60 batches of 1k images
    X = [float(hcat(vec.(imgs)...)) for imgs in partition(imgs, 1000)] |> gpu
    X
end

X = prepare_train();

N = 32 # Size of the encoding
# leakyrelu currently can't be exported by FluxJS when used inside a Chain
encoder = Chain(Dense(28^2, 4*N, relu), Dense(4*N, N, relu)) |> gpu
decoder = Chain(Dense(N, 4*N, relu), Dense(4*N, 28^2)) |> gpu

m = Chain(encoder, decoder)
loss(x) = mse(m(x), x)

# Looking at the loss is booring -- let's use a visual callback function
img(x::Vector) = Gray.(reshape(clamp.(x, 0, 1), 28, 28))
evalcb = 
    throttle(() -> display(
       hcat([vcat(img(X[1][:,i]), img(m(X[1][:,i]).data)) 
                for i in rand(collect(1:1000), 6)]...)
    ), 30)
opt = ADAM(params(m))
# opt = SGD(params(m), 0.1)
@epochs 50 Flux.train!(loss, zip(X), opt, cb = evalcb)

