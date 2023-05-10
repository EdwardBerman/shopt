function generate_heatmap(data::Array{T, 2}, t, cbmin, cbmax) where T<:Real
     heatmap(data,
             aspect_ratio=:equal,  # ensure square cells
             color=:winter,  # use the Winter color map
             cbar=:true,  # add the color bar
             xlabel="u",
 
             ylabel="v",
             c=:log,
             clims=(cbmin, cbmax),  # set the color limits
             title= t,
             xlims=(0.5, size(data, 2) + 0.5),  # set the x-axis limits to include the full cells
             ylims=(0.5, size(data, 1) + 0.5),  # set the y-axis limits to include the full cells
             ticks=:none,  # remove the ticks
             frame=:box,  # draw a box around the plot
             grid=:none  # remove the grid lines
            )
end

#Now we write a function that plots all of the learned parameters with error bars
#errorBars::Array{T, 1}
#yerr=errorBars
function error_plot(learnedParameters::Array{T, 1}, trueParameters::Array{T, 1},errorBars::Array{T, 1}, Title) where T<:Real 
  #Plot the learned parameters
  plot(learnedParameters, 
       label="True Parameters",
       xlabel="Parameter", 
       ylabel="Value", 
       title=Title, 
       seriestype=:scatter, 
       legend=:topleft)
  #labels = ["s", "g1", "g2"]
  #annotate!([labels])
  #Plot the true parameters
  plot!(trueParameters, 
        yerr=errorBars,
        seriestype=:scatter, 
        label="Learned Parameters")
  #Save the plot
  savefig(joinpath("outdir", replace(Title*".png", r"\s+" => "")))
end

function hist(x::Array{T, 1}, y::Array{T, 1}) where T<:Real
  bin_edges = range(-1, stop=1, step=0.1)
  histogram(x, 
            alpha=0.5, 
            bins=bin_edges, 
            label="g1 model", 
            xticks = -1:0.1:1)
  histogram!(y, 
             alpha=0.5, 
             bins=bin_edges, 
             label="g1 data",
             xticks = -1:0.1:1)
  savefig(joinpath("outdir", "histograms_plot.png"))
end

function plot_hm()
  hm = generate_heatmap(star, "I(u,v) Model", amin, amax)
  hm2 = generate_heatmap(costSquaredError, "SquaredError(u,v)", csemin, csemax)
  hm3 = generate_heatmap(starData, "I(u,v) Data", amin, amax)
  hm4 = generate_heatmap(Residuals, "Residuals", rmin, rmax)
  hm5 = generate_heatmap(chiSquare, "ChiSquare", csmin, csmax)
  hm6 = generate_heatmap(pg, "Pixel Grid", amin, amax)
  hm7 = generate_heatmap((star - pg).^2, "Pixel Grid Squared Error", rpgmin, rpgmax)
  savefig(plot(hm, hm3, hm5, hm2, hm4, layout = (2,3),size = (900,400)), joinpath("outdir", "ellipticalGaussianResults.png"))
  savefig(plot(hm, hm6, hm7, layout = (1,3),size = (900,400)), joinpath("outdir", "pixelGrid.png"))
end
