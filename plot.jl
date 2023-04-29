function generate_heatmap(data::Array{T, 2}, t, cbmin, cbmax) where T<:Real
     heatmap(data,
             aspect_ratio=:equal,  # ensure square cells
             color=:winter,  # use the Winter color map
             cbar=:true,  # add the color bar
             xlabel="u",
 
             ylabel="v",
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

function plot_all()
  hm = generate_heatmap(star, "I(u,v) Model", amin, amax)
  savefig(hm, joinpath("outdir", "intensityHeatmap.png"))
  hm2 = generate_heatmap(costSquaredError, "SquaredError(u,v)", csemin, csemax)
  savefig(hm2, joinpath("outdir", "squaredError.png"))
  hm3 = generate_heatmap(starData, "I(u,v) Data", amin, amax)
  savefig(hm3, joinpath("outdir", "intensityHeatmapComparison.png"))
  hm4 = generate_heatmap(Residuals, "Residuals", rmin, rmax)
  savefig(hm4, joinpath("outdir", "residuals.png"))
  hm5 = generate_heatmap(chiSquare, "ChiSquare", csmin, csmax)
  savefig(hm5, joinpath("outdir", "chiSquare.png"))
  savefig(plot(hm, hm3, hm5, hm2, hm4, layout = (2,3),size = (900,400)), joinpath("outdir", "EllipticalGaussianResults.png"))
end