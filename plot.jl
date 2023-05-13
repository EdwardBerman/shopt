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

function error_plot(model, learned, errorModel, errorLearned, t)
  # Generate example data
  x = [1, 5, 10]  # Three points on the x-axis
  labels = ["s", "g1", "g2"]  # Labels for the points
  model_data = model
  learned_data = learned
  errM = errorModel
  errD = errorLearned

  # Plot scatter plot with error bars
  scatter(x, model_data, yerr=errM, label="Model Data", color="blue", alpha=0.6)
  scatter!(x, learned_data, yerr=errD, label="Learned Data", color="red", markersize=4)

  # Set labels for the x-axis points
  xticks!((x, labels))

  title!(t)

  # Show the plot
  plot!(legend=:topright)
end

function plot_err()
  s_modelClean = remove_outliers(s_model)
  g1_modelClean = remove_outliers(g1_model)
  g2_modelClean = remove_outliers(g2_model)

  n_s = size(s_modelClean, 1)
  n_g1 = size(g1_modelClean, 1)
  n_g2 = size(g2_modelClean, 1)

  s_dataClean = remove_outliers(s_data)
  g1_dataClean = remove_outliers(g1_data)
  g2_dataClean = remove_outliers(g2_data)

  n_sD = size(s_modelClean, 1)
  n_g1D = size(g1_modelClean, 1)
  n_g2D = size(g2_modelClean, 1)

  #Plotting Error True Vs Learned
  ps1 = error_plot([s_model, g1_model, g2_model],
             [s_data, g1_data, g2_data],
             [std(s_model)/sqrt(itr), std(g1_model)/sqrt(itr), std(g2_model)/sqrt(itr)],
             [std(s_data)/sqrt(itr), std(g1_data)/sqrt(itr), std(g2_data)/sqrt(itr)],
             "Learned vs True Parameters")
  ps2 = error_plot([s_modelClean, g1_modelClean, g2_modelClean],
             [s_dataClean, g1_dataClean, g2_dataClean],
             [std(s_model)/sqrt(n_s), std(g1_model)/sqrt(n_g1), std(g2_model)/sqrt(n_g2)],
             [std(s_data)/sqrt(n_sD), std(g1_data)/sqrt(n_g1D), std(g2_data)/sqrt(n_g2D)],
             "Learned vs True Parameters Outliers Removed")
  savefig(ps1, joinpath("outdir", "parameterSpread.png"))
  savefig(ps2, joinpath("outdir", "parameterSpreadOutliersRM.png"))
end

function hist(x::Array{T, 1}, y::Array{T, 1}, t1, t2) where T<:Real
  bin_edges = range(-1, stop=1, step=0.1)
  histogram(x, 
            alpha=0.5, 
            bins=bin_edges, 
            label=t1, 
            xticks = -1:0.1:1)
  histogram!(y, 
             alpha=0.5, 
             bins=bin_edges, 
             label=t2,
             xticks = -1:0.1:1)
end
function plot_hist()
  hist1 = hist(s_model, s_data, "s Model", "s Data")
  hist2 = hist(g1_model, g1_data, "g1 Model", "g1 Data")
  hist3 = hist(g2_model, g2_data, "g2 Model", "g2 Data")
  savefig(plot(hist1, hist2, hist3, layout = (1,3)), joinpath("outdir", "histograms_plot.png"))
end

function plot_hm(p)
  hm = generate_heatmap(star, "I(u,v) Model", amin, amax)
  hm2 = generate_heatmap(costSquaredError, "SquaredError(u,v)", csemin, csemax)
  hm3 = generate_heatmap(starData, "I(u,v) Data", amin, amax)
  hm4 = generate_heatmap(Residuals, "Residuals", rmin, rmax)
  hm5 = generate_heatmap(chiSquare, "Chi-Square Residuals, p = "*p, csmin, csmax)
  hm6 = generate_heatmap(pg, "Pixel Grid", amin, amax)
  hm7 = generate_heatmap((star - pg).^2, "Pixel Grid Squared Error", rpgmin, rpgmax)
  s1 = surface(star, colorbar = false)
  s2 = surface(starData, colorbar = false)
  s3 = surface(Residuals, colorbar = false)
  #l1 = generate_heatmap(logStar, "I(u,v) Model Log Plot", amin, amax)
  #l2 = generate_heatmap(logStarData, "I(u,v) Data Log Plot", amin, amax)
  savefig(plot(hm, hm3, hm5, hm2, hm4, layout = (2,3),size = (900,400)), joinpath("outdir", "ellipticalGaussianResults.png"))
  savefig(plot(hm, hm6, hm7, layout = (1,3),size = (900,400)), joinpath("outdir", "pixelGrid.png"))
  savefig(plot(s1, s2, s3, layout = (1,3)), joinpath("outdir", "3dGraphics.png"))
  #savefig(plot(l1, l2, hm5, layout = (1,3)), joinpath("outdir", "logPlots.png"))
end
