function meshgrid(x, y)
    X = [i for i in x, j in 1:length(y)]
    Y = [j for i in 1:length(x), j in y]
    return X, Y
end

function ks(g1Map, g2Map)
  x, y = size(g1Map)
  k1, k2 = meshgrid(fftfreq(y), fftfreq(x))

  g1_hat = fft(g1Map)
  g2_hat = fft(g2Map)

  p1 = k1 * k1 - k2 * k2
  p2 = 2 * k1 * k2
  k2 = k1 * k1 + k2 * k2
  k2[1,1] = 1

  kEhat = (p1 * g1_hat + p2 * g2_hat) / k2
  kBhat = -(p2 * g1_hat - p1 * g2_hat) / k2

  kE = real.(ifft(kEhat))
  kB = real.(ifft(kBhat))

  return kE, kB
end

scale = 1/0.29
ks93, k0 = ks(rand(10,10),rand(10,10))
ksCosmos = imfilter(ks93, Kernel.gaussian(scale))
kshm = Plots.heatmap(ksCosmos, 
                      title="Kaisser-Squires", 
                      xlabel="u", 
                      ylabel="v",
                      xlims=(0.5, size(ksCosmos, 2) + 0.5),  # set the x-axis limits to include the full cells
                      ylims=(0.5, size(ksCosmos, 1) + 0.5),  # set the y-axis limits to include the full cells
                      aspect_ratio=:equal,
                      ticks=:none,  # remove the ticks
                      frame=:box,  # draw a box around the plot
                      grid=:none,  # remove the grid lines
                      size=(1920,1080))

Plots.savefig(kshm, joinpath("outdir","ks.png"))
