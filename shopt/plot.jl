#=
Julia Plots
=#

function error_plot(model, learned, errorModel, errorLearned, t)
  # Generate example data
  x = [1, 5, 9]  # Three points on the x-axis
  labels = ["s", "g1", "g2"]  # Labels for the points
  model_data = model
  learned_data = learned
  errM = errorModel 
  errD = errorLearned 

  # Plot scatter plot with error bars
  Plots.scatter(x, model_data, yerr=errM, label="Model Data", color="blue", alpha=0.6)
  Plots.xticks!(x, labels)
  Plots.scatter!(x, learned_data, yerr=errD, label="Learned Data", color="red", markersize=4)
  Plots.xticks!(x, labels)

  Plots.title!(t)

  # Show the plot
  Plots.plot!(legend=:topright, titlefontsize=30, xguidefontsize=30, yguidefontsize=30, margin = 25mm)
end

function plot_err(s_model=s_model, g1_model=g1_model, g2_model=g2_model, s_data=s_data, g1_data=g1_data, g2_data=g2_data, parametersScatterplot=parametersScatterplot)

  n = size(s_model, 1)
  if parametersScatterplot
    ps1 = error_plot([mean(s_model), mean(g1_model), mean(g2_model)],
                     [mean(s_data), mean(g1_data), mean(g2_data)],
                     [std(s_model)/sqrt(n), std(g1_model)/sqrt(n), std(g2_model)/sqrt(n)],
                     [std(s_data)/sqrt(n), std(g1_data)/sqrt(n), std(g2_data)/sqrt(n)],
                     "Learned vs True Parameters")
    Plots.savefig(Plots.plot(ps1, size = (1920, 1080)), "parametersScatterplot.pdf")
    Plots.savefig(Plots.plot(ps1, size = (1920, 1080)), "parametersScatterplot.png")
  end
end

function hist_1(x::Array{T, 1}, y::Array{T, 1}, t1, t2) where T<:Real
  bin_edges = range(-1, stop=1, step=0.1)
  Plots.histogram(x, 
                  alpha=0.5, 
                  bins=bin_edges,
                  label=t1, 
                  titlefontsize=30, 
                  xguidefontsize=30, 
                  yguidefontsize=30)
  Plots.histogram!(y, 
                   alpha=0.5, 
                   bins=bin_edges, 
                   label=t2,
                   titlefontsize=30, 
                   xguidefontsize=30, 
                   yguidefontsize=30)
  Plots.plot!(margin=15mm)
end
function hist_2(x::Array{T, 1}, y::Array{T, 1}, t1, t2) where T<:Real
  bin_edges = range(-1, stop=1, step=0.1)
  Plots.histogram(x, 
                  alpha=0.5, 
                  bins=bin_edges,
                  label=t1, 
                  xticks = -1:0.1:1,
                  titlefontsize=30, 
                  xguidefontsize=30, 
                  yguidefontsize=30)
  Plots.histogram!(y, 
                   alpha=0.5, 
                   bins=bin_edges, 
                   label=t2,
                   xticks = -1:0.1:1,
                   titlefontsize=30, 
                   xguidefontsize=30, 
                   yguidefontsize=30)
  Plots.plot!(margin=15mm)
end

function plot_hist(s_model=s_model, g1_model=g1_model, g2_model=g2_model, s_data=s_data, g1_data=g1_data, g2_data=g2_data, hist_1=hist_1, hist_2=hist_2, parametersHistogram=parametersHistogram)
  if parametersHistogram
    hist1 = hist_1(s_model, s_data, "s Model", "s Data")
    hist2 = hist_2(g1_model, g1_data, "g1 Model", "g1 Data")
    hist3 = hist_2(g2_model, g2_data, "g2 Model", "g2 Data")
    bin_edges = range(-1, stop=1, step=0.1)
    hist4 = Plots.histogram(s_model - s_data,
                            alpha=0.5, 
                            label="S Model and Data Residuals")
    hist5 = Plots.histogram(g1_model - g1_data, 
                            alpha=0.5, 
                            closed =:both,
                            bins=range(-2, stop=2, step=0.1), 
                            label="g1 Model and Data Residuals", 
                            xticks = -1:0.1:1)
    hist6 = Plots.histogram(g2_model - g2_data, 
                            alpha=0.5,
                            closed =:both,
                            bins=range(-2, stop=2, step=0.1), 
                            label="g2 Model and Data Residuals", 
                            xticks = -1:0.1:1)

    Plots.savefig(Plots.plot(hist1, 
                             hist2, 
                             hist3, 
                             hist4, 
                             hist5, 
                             hist6, 
                             layout = (2,3), 
                             size = (1920,1080)), 
                                "parametersHistogram.pdf")
    
    Plots.savefig(Plots.plot(hist1, 
                             hist2, 
                             hist3, 
                             hist4, 
                             hist5, 
                             hist6, 
                             layout = (2,3), 
                             size = (1920,1080)), 
                                "parametersHistogram.png")
  end
end

function generate_heatmap(data::Array{T, 2}, cbmin, cbmax) where T<:Real
  Plots.heatmap(data,
                aspect_ratio=:equal,  # ensure square cells
                color=:winter,  # use the Winter color map
                cbar=:true,  # add the color bar
                xlabel="u",
                ylabel="v",
                clims=(cbmin, cbmax),  # set the color limits
                xlims=(0.5, size(data, 2) + 0.5),  # set the x-axis limits to include the full cells
                ylims=(0.5, size(data, 1) + 0.5),  # set the y-axis limits to include the full cells
                ticks=:none,  # remove the ticks
                frame=:box,  # draw a box around the plot
                grid=:none,  # remove the grid lines
                titlefontsize=30, 
                xguidefontsize=30, 
                yguidefontsize=30,
                margin = 25mm
                )
end

function generate_heatmap_sp_t(data::Array{T, 2}, t, cbmin, cbmax) where T<:Real
  Plots.heatmap(data,
                aspect_ratio=:equal,  # ensure square cells
                color=:winter,  # use the Winter color map
                cbar=:true,  # add the color bar
                title = t,
                clims=(cbmin, cbmax),  # set the color limits
                xlims=(0.5, size(data, 2) + 0.5),  # set the x-axis limits to include the full cells
                ylims=(0.5, size(data, 1) + 0.5),  # set the y-axis limits to include the full cells
                ticks=:none,  # remove the ticks
                frame=:box,  # draw a box around the plot
                grid=:none,  # remove the grid lines
                margin = 15mm
                )
end

function generate_heatmap_sp_t2(data::Array{T, 2}, t, cbmin, cbmax) where T<:Real
  Plots.heatmap(data,
                aspect_ratio=:equal,  # ensure square cells
                color=:winter,  # use the Winter color map
                cbar=:true,  # add the color bar
                title = t,
                clims=(cbmin, cbmax),  # set the color limits
                xlims=(0.5, size(data, 2) + 0.5),  # set the x-axis limits to include the full cells
                ylims=(0.5, size(data, 1) + 0.5),  # set the y-axis limits to include the full cells
                ticks=:none,  # remove the ticks
                frame=:box,  # draw a box around the plot
                grid=:none,  # remove the grid lines
                margin = 15mm,
                size=(1920, 1080)
                )
end

function generate_heatmap_sp(data::Array{T, 2}, cbmin, cbmax) where T<:Real
  Plots.heatmap(data,
                aspect_ratio=:equal,  # ensure square cells
                color=:winter,  # use the Winter color map
                cbar=:true,  # add the color bar
                clims=(cbmin, cbmax),  # set the color limits
                xlims=(0.5, size(data, 2) + 0.5),  # set the x-axis limits to include the full cells
                ylims=(0.5, size(data, 1) + 0.5),  # set the y-axis limits to include the full cells
                ticks=:none,  # remove the ticks
                frame=:box,  # draw a box around the plot
                grid=:none,  # remove the grid lines
                margin = 15mm
                )
end

function generate_heatmap_titled(data::Array{T, 2}, t, cbmin, cbmax) where T<:Real
  Plots.heatmap(data,
                aspect_ratio=:equal,  # ensure square cells
                color=:winter,  # use the Winter color map
                cbar=:true,  # add the color bar
                xlabel="u",
                ylabel="v",
                title = t,
                clims=(cbmin, cbmax),  # set the color limits
                xlims=(0.5, size(data, 2) + 0.5),  # set the x-axis limits to include the full cells
                ylims=(0.5, size(data, 1) + 0.5),  # set the y-axis limits to include the full cells
                ticks=:none,  # remove the ticks
                frame=:box,  # draw a box around the plot
                grid=:none,  # remove the grid lines
                titlefontsize=30, 
                xguidefontsize=30, 
                yguidefontsize=30,
                margin = 25mm
                )
end

#=
function plot_hm(r=r, c=c, outdir=outdir, sampled_indices=sampled_indices, starCatalog=starCatalog, pixelGridFits=pixelGridFits, get_middle_15x15=get_middle_15x15, generate_heatmap_sp_t=generate_heatmap_sp_t, generate_heatmap_sp=generate_heatmap_sp,generate_heatmap_titled=generate_heatmap_titled)#p
  star1 = sampled_indices[1]
  star2 = sampled_indices[2]
  star3 = sampled_indices[3]
  #p_value = p
  chiSquareTemplate = zeros(r, c)
  chiSquare = []
  
  for i in 1:3
    starCatalog[sampled_indices[i]] = Float64.(starCatalog[sampled_indices[i]])
  end

  for i in 1:3
    chiSquareTemplate = Float64.(starCatalog[sampled_indices[i]]) - pixelGridFits[sampled_indices[i]].^2
    chiSquareTemplate = chiSquareTemplate./(errVignets[sampled_indices[i]].^2)
    #err cutout , divide matrices pointwise
    push!(chiSquare, chiSquareTemplate)                              
  end


  hm11 = generate_heatmap_sp_t(get_middle_15x15(starCatalog[star1]), 
                               "Model",
                               minimum([minimum(starCatalog[star1]), minimum(pixelGridFits[star1])]), 
                               maximum([maximum(starCatalog[star1]), maximum(pixelGridFits[star1])]))
  hm12 = generate_heatmap_sp_t(get_middle_15x15(pixelGridFits[star1]), 
                               "Data", 
                               minimum([minimum(starCatalog[star1]), minimum(pixelGridFits[star1])]), 
                               maximum([maximum(starCatalog[star1]), maximum(pixelGridFits[star1])]))
  hm13 = generate_heatmap_sp_t(get_middle_15x15(chiSquare[1]), 
                               "ChiSquare", 
                               minimum(chiSquare[1]), 
                               maximum(chiSquare[1]))
  
  hm21 = generate_heatmap_sp(get_middle_15x15(starCatalog[star2]), 
                             minimum([minimum(starCatalog[star2]), minimum(pixelGridFits[star2])]), 
                             maximum([maximum(starCatalog[star2]), maximum(pixelGridFits[star2])]))

  hm22 = generate_heatmap_sp(get_middle_15x15(pixelGridFits[star2]), 
                             minimum([minimum(starCatalog[star2]), minimum(pixelGridFits[star2])]), 
                             maximum([maximum(starCatalog[star2]), maximum(pixelGridFits[star2])]))
  
  hm23 = generate_heatmap_sp(get_middle_15x15(chiSquare[2]), 
                             minimum(chiSquare[2]), 
                             maximum(chiSquare[2]))

  hm31 = generate_heatmap_sp(get_middle_15x15(starCatalog[star3]), 
                             minimum([minimum(starCatalog[star3]), minimum(pixelGridFits[star3])]), 
                             maximum([maximum(starCatalog[star3]), maximum(pixelGridFits[star3])]))
  
  hm32 = generate_heatmap_sp(get_middle_15x15(pixelGridFits[star3]), 
                             minimum([minimum(starCatalog[star3]), minimum(pixelGridFits[star3])]), 
                             maximum([maximum(starCatalog[star3]), maximum(pixelGridFits[star3])]))
  
  hm33 = generate_heatmap_sp(get_middle_15x15(chiSquare[3]), 
                             minimum(chiSquare[3]), 
                             maximum(chiSquare[3]))
#=
  hm41 = generate_heatmap_sp(starCatalog[star4], minimum([minimum(starCatalog[star1]), minimum(pixelGridFits[star4])]), maximum([maximum(starCatalog[star4]), maximum(pixelGridFits[star4])]))
  hm42 = generate_heatmap_sp(pixelGridFits[star4], minimum([minimum(starCatalog[star4]), minimum(pixelGridFits[star4])]), maximum([maximum(starCatalog[star4]), maximum(pixelGridFits[star4])]))
  hm43 = generate_heatmap_sp(chiSquare[4], minimum(chiSquare[4]), maximum(chiSquare[4]))
  
  hm51 = generate_heatmap_sp(starCatalog[star5], minimum([minimum(starCatalog[star5]), minimum(pixelGridFits[star5])]), maximum([maximum(starCatalog[star5]), maximum(pixelGridFits[star5])]))
  hm52 = generate_heatmap_sp(pixelGridFits[star5], minimum([minimum(starCatalog[star5]), minimum(pixelGridFits[star5])]), maximum([maximum(starCatalog[star5]), maximum(pixelGridFits[star5])]))
  hm53 = generate_heatmap_sp(chiSquare[5], minimum(chiSquare[5]), maximum(chiSquare[5]))

  hm61 = generate_heatmap_sp(starCatalog[star6], minimum([minimum(starCatalog[star6]), minimum(pixelGridFits[star6])]), maximum([maximum(starCatalog[star6]), maximum(pixelGridFits[star6])]))
  hm62 = generate_heatmap_sp(pixelGridFits[star6], minimum([minimum(starCatalog[star6]), minimum(pixelGridFits[star6])]), maximum([maximum(starCatalog[star6]), maximum(pixelGridFits[star6])]))
  hm63 = generate_heatmap_sp(chiSquare[6], minimum(chiSquare[6]), maximum(chiSquare[6]))
=#
  #=
  fft_image1 = fft(complex.(starCatalog[star1] .- pixelGridFits[star1]))
  fft_image1 = abs2.(fft_image1)

  pk1 = []
  for i in 1:10
    radius = range(1, max(r/2, c/2) - 1, length=10)
    push!(pk1, powerSpectrum(fft_image1, radius[i]))
  end

  fftmin1 = minimum(fft_image1)       
  fftmax1 = maximum(fft_image1) 
  
  fft_image2 = fft(complex.(starCatalog[star2] .- pixelGridFits[star2]))
  fft_image2 = abs2.(fft_image2)

  pk2 = []
  for i in 1:10
    radius = range(1, max(r/2, c/2) - 1, length=10)
    push!(pk2, powerSpectrum(fft_image2, radius[i]))
  end

  fftmin2 = minimum(fft_image2)       
  fftmax2 = maximum(fft_image2) 
  
  fft_image3 = fft(complex.(starCatalog[star3] .- pixelGridFits[star3]))
  fft_image3 = abs2.(fft_image3)

  pk3 = []
  for i in 1:10
    radius = range(1, max(r/2, c/2) - 1, length=10)
    push!(pk3, powerSpectrum(fft_image3, radius[i]))
  end

  fftmin3 = minimum(fft_image3)       
  fftmax3 = maximum(fft_image3) 

  hm8 = generate_heatmap_sp_t(fft_image1, "FFT Residuals", fftmin1, fftmax1)
  hm9 = generate_heatmap_sp(fft_image2, fftmin2, fftmax2)
  hm10 = generate_heatmap_sp(fft_image3, fftmin3, fftmax3)
  =#
  zmn = minimum([minimum(starCatalog[star3]), minimum(pixelGridFits[star3])])
  zmx = maximum([maximum(starCatalog[star3]), maximum(pixelGridFits[star3])])

  s1 = Plots.surface(title="Model Star", titlefontsize=30, starCatalog[star3], zlim=(zmn, zmx), colorbar = false)
  s2 = Plots.surface(title="pixelGridFit", titlefontsize=30, pixelGridFits[star3], zlim=(zmn, zmx), colorbar = false)
  s3 = Plots.surface(title="Residuals", titlefontsize=30, starCatalog[star3] .- pixelGridFits[star3], zlim=(2*zmn, 2*zmx), colorbar = false)

  meanRelativeError = zeros(r,c)
  for j in 1:size(starCatalog[star1], 1)
    for k in 1:size(starCatalog[star1], 2)
      RelativeError = []
      for i in 1:length(starCatalog)
        push!(RelativeError, abs.(starCatalog[i][j,k] .- pixelGridFits[i][j,k]) ./ abs.(starCatalog[i][j,k] .+ 1e-10))
      end
      meanRelativeError[j,k] = mean(RelativeError)
    end
  end

  RelativeErrorHM = generate_heatmap_sp_t2(meanRelativeError, "Mean Relative Error at Each Pixel", minimum(meanRelativeError), maximum(meanRelativeError))



  pk_1 = Plots.plot(pk1, 
                    xlabel = "k", 
                    linewidth=5, 
                    tickfontsize=16,
                    titlefontsize=30, 
                    xguidefontsize=30, 
                    yguidefontsize=30, 
                    ylabel = "P(k)", 
                    title = "Power Spectrum", 
                    margin = 25mm)
  pk_2 = Plots.plot(pk2,
                    xlabel = "k", 
                    linewidth=5, 
                    tickfontsize=16,
                    titlefontsize=30, 
                    xguidefontsize=30, 
                    yguidefontsize=30, 
                    ylabel = "P(k)", 
                    margin = 25mm)
  pk_3 = Plots.plot(pk3,
                    xlabel = "k", 
                    linewidth=5, 
                    tickfontsize=16,
                    titlefontsize=30, 
                    xguidefontsize=30, 
                    yguidefontsize=30, 
                    ylabel = "P(k)", 
                    margin = 25mm)
  
  #title1 = Plots.plot(title = "I(u,v) Model", titlefontsize=30, grid = false, showaxis = false, bottom_margin = -25Plots.px)
  #title2 = Plots.plot(title = "I(u,v) Data", titlefontsize=30, grid = false, showaxis = false, bottom_margin = -25Plots.px)
  #title3 = Plots.plot(title = "Chi-Square Residuals", titlefontsize=30, grid = false, showaxis = false, bottom_margin = -25Plots.px)

  Plots.savefig(Plots.plot(hm11, hm12, hm13, 
                           hm21, hm22, hm23, 
                           hm31, hm32, hm33, 
                              layout = (3,3),
                              widths=ones(9),
                              size = (1920,1080)), 
                                  joinpath("outdir", "pixelGridFit.pdf"))
  
  Plots.savefig(Plots.plot(hm11, hm12, hm13, 
                           hm21, hm22, hm23, 
                           hm31, hm32, hm33, 
                           layout = (3,3),
                           widths=ones(9),
                              size = (1920,1080)), 
                                  joinpath("outdir", "pixelGridFit.png"))

  Plots.savefig(RelativeErrorHM, joinpath("outdir", "relativeErrorHM.pdf"))
  Plots.savefig(RelativeErrorHM, joinpath("outdir", "relativeErrorHM.png"))
  Plots.savefig(Plots.plot(s1, 
                           s2, 
                           s3, 
                           layout = (1,3),
                           size = (1920,1080)), 
                           joinpath(outdir, "3dPixelGridFit.png"))
  
  Plots.savefig(Plots.plot(pk_1, 
                           pk_2,
                           pk_3,
                            layout = (3,1),
                            size = (1920,1080)), 
                                joinpath(outdir, "powerspectra.png"))
  
  Plots.savefig(Plots.plot(hm8,  
                           hm9, 
                           hm10,
                            layout = (3,1),
                            size = (1920,1080)), 
                                joinpath(outdir, "fftresiduals.png"))
  Plots.savefig(Plots.plot(pk_1, 
                           pk_2,
                           pk_3,
                            layout = (3,1),
                            size = (1920,1080)), 
                                joinpath(outdir, "powerspectra.pdf"))
  
  Plots.savefig(Plots.plot(hm8,  
                           hm9, 
                           hm10,
                            layout = (3,1),
                            size = (1920,1080)), 
                                joinpath(outdir, "fftresiduals.pdf"))

end
=#
#=
Revisit Later?
  
=#
#=
#=
661   scale = 1/0.29
662   ks93, k0 = ks(g1_map, g2_map)
663   ksCosmos = get_middle_15x15(imfilter(ks93, Kernel.gaussian(scale)))
664   kshm = Plots.heatmap(ksCosmos,
665                        title="Kaisser-Squires", 
666                        xlabel="u", 
667                        ylabel="v",
668                        xlims=(0.5, size(ksCosmos, 2) + 0.5),  # set the x-axis limits to include the full cells
669                        ylims=(0.5, size(ksCosmos, 1) + 0.5),  # set the y-axis limits to include the full cells
670                        aspect_ratio=:equal,
671                        ticks=:none,  # remove the ticks
672                        frame=:box,  # draw a box around the plot
673                        grid=:none,  # remove the grid lines
674                        size=(1920,1080))
675 
676   Plots.savefig(kshm, joinpath("outdir","kaisserSquires.png"))
677   =#  
=#
