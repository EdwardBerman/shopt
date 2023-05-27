using CairoMakie, Distributions
using Random
Random.seed!(1234)
d = Normal()
b = Binomial(15, 0.7)
n = 350
scatter(rand(d,n), rand(b, n);
    markersize = 12*abs.(rand(d, n)),
    color = tuple.(:orangered, rand(n)),
    strokewidth = 0.5,
    strokecolor = :white,
    axis = (;
        xlabel = "x", ylabel = "y"),
    figure = (;
        resolution = (600,400))
    );
