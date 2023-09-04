# ---------------------------------------------------------#
@time begin
  include("argparser.jl")
  include("fancyPrint.jl")

  try
    process_arguments(ARGS)
  catch err
    println("Error: ", err)
    println("Usage: julia shopt.jl <configdir> <outdir> <catalog>")
    exit(1)
  end

  configdir = ARGS[1]
  outdir = ARGS[2]
  catalog = ARGS[3]

  if isdir(outdir)
    println("━ Outdir found")
  else
    println("━ Outdir not found, creating...")
    mkdir(outdir)
  end
end

# ---------------------------------------------------------#
fancyPrint("Handling Imports")
@time begin
  using Base: initarray!
  using YAML
  using BenchmarkTools
  using Plots
  using ForwardDiff
  using LinearAlgebra
  using PyCall
  using Random
  using Distributions
  using SpecialFunctions
  using Optim
  using IterativeSolvers
  using QuadGK
  using DataFrames
  using FFTW
  using CSV
  using Images, ImageFiltering
  using ImageTransformations
  using Measures
  using ProgressBars
  using UnicodePlots
  using Flux
  using Flux.Optimise
  using Flux.Losses
  using Flux: onehotbatch, throttle, mse, msle, mae
  using CairoMakie
  using Dates
  using MultivariateStats
  using Base.Threads
  #using Interpolations
end
println("━ Start Time ", Dates.now())
start = Dates.now()
# ---------------------------------------------------------#
fancyPrint("Reading .jl Files")
@time begin
  include("plot.jl")
  include("analyticLBFGS.jl")
  include("radialProfiles.jl")
  include("masks.jl")
  include("outliers.jl")
  include("dataOutprocessing.jl")
  include("powerSpectrum.jl")
  include("kaisserSquires.jl")
  include("dataPreprocessing.jl")
  include("interpolate.jl")
  include("pixelGridAutoencoder.jl")
  include("pca.jl")
  include("chisq.jl")
  include("reader.jl")
  #include("lk.jl")
end
# ---------------------------------------------------------#
#fancyPrint("Running Source Extractor")
# ---------------------------------------------------------#

fancyPrint("Processing Data for Fit")
@time begin
  
  if mode == "chisq"
    starCatalog, r, c, itr, u_coordinates, v_coordinates, outlier_indices, errVignets = cataloging(ARGS)
  else
    starCatalog, r, c, itr, u_coordinates, v_coordinates, outlier_indices = cataloging(ARGS)
  end
  #starCatalog = starCatalog
  #errVignets = errVignets
  #u_coordinates = u_coordinates
  #v_coordinates = v_coordinates
  itr = length(starCatalog)
  
  
  #=
  starCatalog, r,c, itr = catalogingWEBBPSF()
  u_coordinates = rand(2)
  v_coordinates = rand(2)

  itr = length(starCatalog)
  =#

  #=
  starCatalog, r, c, u_coordinates, v_coordinates = gridPSFS() #return catalogNew, rows, cols, u_coords, v_coords
  errVignets = starCatalog
  itr = length(starCatalog)
  =#
end

starData = zeros(r, c)

#A_model = zeros(itr)
s_model = zeros(itr)
g1_model = zeros(itr)
g2_model = zeros(itr)

#A_data = zeros(itr)
s_data = zeros(itr)
g1_data = zeros(itr)
g2_data = zeros(itr)

failedStars = []
# ---------------------------------------------------------#
fancyPrint("Analytic Profile Fit for Model Star")
@time begin
  pb = tqdm(1:itr)
  for i in pb
    initial_guess = rand(3) #println("\t initial guess [σ, e1, e2]: ", initial_guess)
    set_description(pb, "Star $i/$itr Complete")
    
    global iteration = i
    try
      global x_cg = optimize(cost, 
                             g!, 
                             initial_guess, 
                             LBFGS(),#ConjugateGradient()
                             Optim.Options(g_tol = minAnalyticGradientModel))#Optim.Options(callback = cb) #Optim.Options(g_tol = 1e-6))
      
      s_model[i] = x_cg.minimizer[1]^2
      e1_guess = x_cg.minimizer[2]
      e2_guess = x_cg.minimizer[3]

      ellipticityData = sqrt((e1_guess)^2 + (e2_guess)^2)
      normGdata = sqrt(1 + 0.5*( (1/ellipticityData^2) - sqrt( (4/ellipticityData^2) + (1/ellipticityData^4)  )  )) 
      ratioData = ellipticityData/normGdata
      g1_model[i] = e1_guess/ratioData            
      g2_model[i] = e2_guess/ratioData  
      
    catch
      println("Star $i failed")
      #push!(failedStars, i)
      s_model[i] = 0
      g1_model[i] = 0
      g2_model[i] = 0
      continue
    end
  end
end

s_blacklist = []
for i in 1:length(s_model)
  if (s_model[i] < sLowerBound || s_model[i] > sUpperBound) #i in failedStars is optional Since Failed Stars are assigned s=0 
    push!(s_blacklist, i)
  end
end

println("\n━ Blacklisted Stars: ", s_blacklist)
println("\n━ Blacklisted $(length(s_blacklist)) stars on the basis of s < $sLowerBound or s > $sUpperBound (Failed Stars Assigned 0)." )
println("\n━ NB: These blacklisted stars are being indexed after the initial removal on the basis of signal to noise, not based off of their original location in the star catalog.")
for i in sort(s_blacklist, rev=true)
  splice!(starCatalog, i)
  #splice!(errVignets, i)
  splice!(s_model, i)
  splice!(g1_model, i)
  splice!(g2_model, i)
  splice!(u_coordinates, i)
  splice!(v_coordinates, i)
end

failedStars = []

# ---------------------------------------------------------#
fancyPrint("Pixel Grid Fit")
pixelGridFits = []

if mode == "chisq"
  println("━  χ2  Mode...\n")
  @time begin
    pb = tqdm(1:itr)
    for i in pb
      set_description(pb, "Star $i/$itr Complete") 
      global iteration = i
      initial_guess = rand(r*c)
      try
        global chisq_cg = optimize(chisq_cost, 
                                   chisq_g!, 
                                   initial_guess, 
                                   LBFGS())#,#ConjugateGradient()
                                   
                                   #Optim.Options(g_tol = chisq_stopping_gradient))#Optim.Options(callback = cb) #Optim.Options(g_tol = 1e-6))
      catch ex
        println(ex)
        println("Star $i failed")
        push!(failedStars, i)
        push!(pixelGridFits, zeros(r,c))
        continue
      end
      if unity_sum
        pgf_current = reshape(chisq_cg.minimizer, (r, c))./sum(reshape(chisq_cg.minimizer, (r, c)))
      else
        pgf_current = reshape(chisq_cg.minimizer, (r, c))
      end
      push!(pixelGridFits, pgf_current)
    end 
  end 
end


if mode == "autoencoder"
  println("━ Autoencoder Mode...\n")
  println(autoencoder,"\n")
  @time begin
    pb = tqdm(1:length(starCatalog))
    for i in pb
      set_description(pb, "Star $i/$(length(starCatalog)) Complete")
      global iteration = i
      
      # Format some random image data
      #data = nanToGaussian(starCatalog[i], s_model[i], g1_model[i], g2_model[i], r/2, c/2)
      #data = reshape(data, length(data))
      data = nanToZero(reshape(starCatalog[i], length(starCatalog[i])))
      #data = Float32.(reshape(nanToZero(starCatalog[i]), r, c, 1, 1))
      
      # Train the autoencoder
      #data_x̂ = sample_image(autoencoder(data_x),r)
      try
        min_gradient = minPixelGradient
        for epoch in 1:epochs
          Flux.train!(loss, Flux.params(autoencoder), [(data, )], optimizer) #loss#Flux.params(autoencoder))

          grad = Flux.gradient(Flux.params(autoencoder)) do
            loss(data)
          end
          grad_norm = norm(grad)
          if (grad_norm < min_gradient) #min_gradient
            #println(epoch)
            break
          end
        end
        # Take a sample input image
        #input_image = reshape(starCatalog[i], length(starCatalog[i]))
    
        # Pass the input image through the autoencoder to get the reconstructed image
        reconstructed_image = autoencoder(data) #autoencoder(input_image)

        if unity_sum
          pgf_current = reshape(reconstructed_image, (r, c))./sum(reshape(reconstructed_image, (r, c)))
        else
          pgf_current = reshape(reconstructed_image, (r, c))
        end
        push!(pixelGridFits, pgf_current)
      catch ex
        println(ex)
        println("Star $i failed")
        push!(failedStars, i)
        push!(pixelGridFits, zeros(r,c))
        continue
      end
    end
  end
end

if mode == "PCA"
  println("━ PCA Mode...")
  @time begin
    pb = tqdm(1:length(starCatalog))
    for i in pb
      set_description(pb, "Star $i/$(length(starCatalog)) Complete")
      global iteration = i
      data = nanToZero(starCatalog[i])
      try
        if unity_sum
          push!(pixelGridFits, smooth(pca_image(data,PCAterms), lanczos)./sum(smooth(pca_image(data,PCAterms), lanczos)))
        else
          push!(pixelGridFits, smooth(pca_image(data,PCAterms), lanczos))
        end
      catch
        println("Star $i failed")
        push!(failedStars, i)
        push!(pixelGridFits, zeros(r,c))
        continue
      end

    end
  end
end



GC.gc()


println("━ failed stars:", failedStars)
# ---------------------------------------------------------#
fancyPrint("Analytic Profile Fit for Learned Star")
#Copy Star Catalog then replace it with the learned pixel grid stars
@time begin
  pb = tqdm(1:length(starCatalog))
  for i in pb
    initial_guess = rand(3) #println("\t initial guess [σ, e1, e2]: ", initial_guess)
    set_description(pb, "Star $i/$(length(starCatalog)) Complete")
    
    global iteration = i
    try 
      global y_cg = optimize(costD, 
                             gD!, 
                             initial_guess,
                             LBFGS(),#ConjugateGradient()ConjugateGradient(),
                             Optim.Options(g_tol = minAnalyticGradientLearned)) #Optim.Options(callback = cb)
    
      s_data[i] = y_cg.minimizer[1]^2
      e1_guess = y_cg.minimizer[2]
      e2_guess = y_cg.minimizer[3]

      ellipticityData = sqrt((e1_guess)^2 + (e2_guess)^2)
      normGdata = sqrt(1 + 0.5*( (1/ellipticityData^2) - sqrt( (4/ellipticityData^2) + (1/ellipticityData^4)  )  )) 
      ratioData = ellipticityData/normGdata
      g1_data[i] = e1_guess/ratioData            
      g2_data[i] = e2_guess/ratioData  
      

      if s_data[i] < sLowerBound || s_data[i] > sUpperBound 
        push!(failedStars, i) 
      end
    catch
      println("Star $i failed")
      s_data[i] = 0
      g1_data[i] = 0
      g2_data[i] = 0
      continue
    end
    #println("\t Found A: ", A_data[i], "\t s: ", s_data[i]^2, "\t g1: ", g1_data[i], "\t g2: ", g2_data[i])

  end
end


println("━ failed stars: ", unique(failedStars))
println("\n━ Rejected $(length(unique(failedStars))) more stars for failing or having either s < $sLowerBound or s > $sUpperBound when fitting an analytic profile to an autoencoded image.")
println("\n━ NB: These failed stars are being indexed after both the screening of signal to noise and the blacklisting of s values.")
failedStars = unique(failedStars)

for i in sort(failedStars, rev=true)
  splice!(pixelGridFits, i)
  splice!(s_data, i)
  splice!(g1_data, i)
  splice!(g2_data, i)
  splice!(s_model, i)
  splice!(g1_model, i)
  splice!(g2_model, i)
  splice!(u_coordinates, i)
  splice!(v_coordinates, i)
  splice!(starCatalog, i)
  #splice!(errVignets, i)
end

GC.gc()

# ---------------------------------------------------------#
fancyPrint("Transforming (x,y) -> (u,v) | Interpolation [s, g1, g2] Across the Field of View")

s_data = s_data[1:length(pixelGridFits)]
g1_data = g1_data[1:length(pixelGridFits)]
g2_data = g2_data[1:length(pixelGridFits)]

s_tuples = []
for i in 1:length(starCatalog)
  push!(s_tuples, (u_coordinates[i], v_coordinates[i], s_data[i]))
end

s_fov = optimize(interpCostS, polyG_s!, rand(10), LBFGS())
sC = s_fov.minimizer
println("\ns(u,v) = \n$(sC[1]) u³ \n+ $(sC[2]) v³ \n+ $(sC[3]) u²v \n+ $(sC[4]) v²u \n+ $(sC[5]) u² \n+ $(sC[6]) v² \n+ $(sC[7]) uv \n+ $(sC[8]) u \n+ $(sC[9]) v \n+ $(sC[10])\n")

s(u,v) = sC[1]*u^3 + sC[2]*v^3 + sC[3]*u^2*v + sC[4]*v^2*u + sC[5]*u^2 + sC[6]*v^2 + sC[7]*u*v + sC[8]*u + sC[9]*v + sC[10]
ds_du(u,v) = sC[1]*3*u^2 + sC[3]*2*u*v + sC[4]*v^2 + sC[5]*2*u + sC[7]*v + sC[8]
ds_dv(u,v) = sC[2]*3*v^2 + sC[3]*u^2 + sC[4]*2*u*v + sC[6]*2*v + sC[7]*u + sC[9]

g1_tuples = []
for i in 1:length(starCatalog)
  push!(g1_tuples, (u_coordinates[i], v_coordinates[i], g1_data[i]))
end

g1_fov = optimize(interpCostg1, polyG_g1!, rand(10), LBFGS())
g1C = g1_fov.minimizer
println("\ng1(u,v) = \n$(g1C[1]) u³ \n+ $(g1C[2]) v³ \n+ $(g1C[3]) u²v \n+ $(g1C[4]) v²u \n+ $(g1C[5]) u² \n+ $(g1C[6]) v² \n+ $(g1C[7]) uv \n+ $(g1C[8]) u \n+ $(g1C[9]) v \n+ $(g1C[10])\n")

g1(u,v) = g1C[1]*u^3 + g1C[2]*v^3 + g1C[3]*u^2*v + g1C[4]*v^2*u + g1C[5]*u^2 + g1C[6]*v^2 + g1C[7]*u*v + g1C[8]*u + g1C[9]*v + g1C[10]
dg1_du(u,v) = g1C[1]*3*u^2 + g1C[3]*2*u*v + g1C[4]*v^2 + g1C[5]*2*u + g1C[7]*v + g1C[8]
dg1_dv(u,v) = g1C[2]*3*v^2 + g1C[3]*u^2 + g1C[4]*2*u*v + g1C[6]*2*v + g1C[7]*u + g1C[9]

g2_tuples = []
for i in 1:length(starCatalog)
  push!(g2_tuples, (u_coordinates[i], v_coordinates[i], g2_data[i]))
end
h_uv_data = g2_tuples

g2_fov = optimize(interpCostg2, polyG_g2!, rand(10), LBFGS())
g2C = g2_fov.minimizer
println("\ng2(u,v) = \n$(g2C[1]) u³ \n+ $(g2C[2]) v³ \n+ $(g2C[3]) u²v \n+ $(g2C[4]) v²u \n+ $(g2C[5]) u² \n+ $(g2C[6]) v² \n+ $(g2C[7]) uv \n+ $(g2C[8]) u \n+ $(g2C[9]) v \n+ $(g2C[10])\n")

g2(u,v) = g2C[1]*u^3 + g2C[2]*v^3 + g2C[3]*u^2*v + g2C[4]*v^2*u + g2C[5]*u^2 + g2C[6]*v^2 + g2C[7]*u*v + g2C[8]*u + g2C[9]*v + g2C[10]
dg2_du(u,v) = g2C[1]*3*u^2 + g2C[3]*2*u*v + g2C[4]*v^2 + g2C[5]*2*u + g2C[7]*v + g2C[8]
dg2_dv(u,v) = g2C[2]*3*v^2 + g2C[3]*u^2 + g2C[4]*2*u*v + g2C[6]*2*v + g2C[7]*u + g2C[9]

#println("\n** Adding a Progress Bar Dramatically Increases the Run Time, but note that Interpolation across the FOV is taking place! **\n")

#PolynomialMatrix = ones(r,c, 10)
  
function sample_indices(array, k)
  indices = collect(1:length(array))  # Create an array of indices
  return randperm(length(indices))[1:k] #sample(indices, k, replace = false)
end

total_samples = length(pixelGridFits)
#training_ratio = 0.8
training_samples = round(Int, training_ratio * total_samples)

training_indices = sample_indices(pixelGridFits, training_samples)
training_stars = pixelGridFits[training_indices]
training_u_coords = u_coordinates[training_indices]
training_v_coords = v_coordinates[training_indices]

validation_indices = setdiff(1:total_samples, training_indices)
validation_stars = pixelGridFits[validation_indices]
validation_u_coords = u_coordinates[validation_indices]
validation_v_coords = v_coordinates[validation_indices]
validation_star_catalog = starCatalog[validation_indices]

PolynomialMatrix = ones(r,c, (degree + 1) * (degree + 2) ÷ 2 )

fancyPrint("Transforming (x,y) -> (u,v) | Interpolation Pixel by Pixel Across the Field of View")

function compute_single_star_reconstructed_value(PolynomialMatrix, x, y, degree)
    r, c, _ = size(PolynomialMatrix)
    reconstructed_star = zeros(r, c)
    for i in 1:r
        for j in 1:c
            p = PolynomialMatrix[i, j, :]
            reconstructed_star[i, j] = objective_function(p, x, y, degree)
        end
    end
    return reconstructed_star
end

function compute_mse(reconstructed_matrix, true_matrix, err_map)
  return mean((reconstructed_matrix .- true_matrix) .^ 2 ./(err_map.^2))
end

function worst_10_percent(errors)
    n = length(errors)
    star_errors = [(i, errors[i]) for i in 1:n]
    sort!(star_errors, by = x->x[2], rev=true)
    threshold_idx = Int(ceil(0.10 * n))
    worst_stars = star_errors[1:threshold_idx]
    return [star[1] for star in worst_stars]
end

@time begin
  global training_stars, training_u_coords, training_v_coords
  for loop in 1:iterationsPolyfit
    #print(length(iterationsPolyfit))
    println("━ Iteration: $loop")
    @threads for i in 1:r
      for j in 1:c
        z_data = [star[i, j] for star in training_stars]
        pC = polynomial_optimizer(degree, training_u_coords, training_v_coords, z_data)
        PolynomialMatrix[i,j,:] .= pC
      end
    end
    #=
    for i in 1:r
      for j in 1:c
        z_data = [star[i, j] for star in training_stars]
        pC = polynomial_optimizer(degree, training_u_coords, training_v_coords, z_data)
        PolynomialMatrix[i,j,:] .= pC
      end
    end
    =#
    
    #training_errors = Threads.@spawn [compute_mse(compute_single_star_reconstructed_value(PolynomialMatrix, training_u_coords[idx], training_v_coords[idx], degree), training_stars[idx]) for idx in 1:length(training_stars)]
    
    training_errors = []
    for idx in 1:length(training_stars)
      reconstructed_star = compute_single_star_reconstructed_value(PolynomialMatrix, training_u_coords[idx], training_v_coords[idx], degree)
      push!(training_errors, compute_mse(reconstructed_star, training_stars[idx], errVignets[idx]))
    end
    
    bad_indices = worst_10_percent(training_errors)
   
    if loop != iterationsPolyfit
      new_training_stars = []
      new_training_u_coords = []
      new_training_v_coords = []
      for i in 1:length(training_stars)
          if i ∉ bad_indices
              push!(new_training_stars, training_stars[i])
              push!(new_training_u_coords, training_u_coords[i])
              push!(new_training_v_coords, training_v_coords[i])
          end
      end
      
      global training_stars = new_training_stars
      println("$(length(new_training_stars)) training stars")
      global training_u_coords = new_training_u_coords
      global training_v_coords = new_training_v_coords
    end
  end
end

#= Future Work for iterative refinement
global itr_count = 1

@time begin    
    global training_stars, training_u_coords, training_v_coords    
    
    # Initialize a flag for the while loop
    global outliers_exist = true    
    while outliers_exist  # Keep iterating as long as there are outliers
        #println("here")
        current_itr = itr_count
        println("━ Iteration $current_itr")
        global itr_count += 1
        @threads for i in 1:r    
            for j in 1:c    
                z_data = [star[i, j] for star in training_stars]    
                pC = polynomial_optimizer(degree, training_u_coords, training_v_coords, z_data)    
                PolynomialMatrix[i,j,:] .= pC    
            end    
        end    
            
        training_errors = []    
        for idx in 1:length(training_stars)    
            reconstructed_star = compute_single_star_reconstructed_value(PolynomialMatrix, training_u_coords[idx], training_v_coords[idx], degree)
            push!(training_errors, compute_mse(reconstructed_star, training_stars[idx]))    
        end    
        
        # Calculate standard deviation of training_errors
        error_std_dev = std(training_errors)    
        
        # Find indices of outliers which have an error greater than 2 times the std deviation
        bad_indices = findall(x -> x > 3 * error_std_dev, training_errors)    
        
        if isempty(bad_indices)  # If no outliers found, set the flag to false
            #println("here")
            global outliers_exist = false
        else
            new_training_stars = []    
            new_training_u_coords = []    
            new_training_v_coords = []    
            for i in 1:length(training_stars)    
                if i ∉ bad_indices    
                    push!(new_training_stars, training_stars[i])    
                    push!(new_training_u_coords, training_u_coords[i])    
                    push!(new_training_v_coords, training_v_coords[i])    
                end    
            end    
            
            global training_stars = new_training_stars 
            println("Number of training stars after iteration $current_itr: ", length(new_training_stars))
            if length(new_training_stars) < 30
              global outliers_exist = false
              println("Training Stars < 30, terminating...")
            end
            
            global training_u_coords = new_training_u_coords    
            global training_v_coords = new_training_v_coords    
        end    
    end    
end  
=#

GC.gc()

try
  global sampled_indices = sort(sample_indices(validation_indices, 3))
catch
  global sampled_indices = rand(3)
end
#=
println("Sampled indices: ", sampled_indices)
meanRelativeError = []
for i in 1:length(starCatalog)
  RelativeError = []
  for j in 1:size(starCatalog[i], 1)
    for k in 1:size(starCatalog[i], 2)
      push!(RelativeError, abs.(starCatalog[i][j,k] .- pixelGridFits[i][j,k]) ./ abs.(starCatalog[i][j,k] .+ 1e-10))
    end
  end
  push!(meanRelativeError, mean(RelativeError))
end
=#

# ---------------------------------------------------------#
fancyPrint("Plotting")
@time begin

  plot_hist()
  plot_err()

  starSample = rand(1:length(starCatalog))
  a = starCatalog[starSample]
  b = pixelGridFits[starSample]

  a = nanToZero(a)
  b = nanToZero(b)

  cmx = maximum([maximum(a), maximum(b)])
  cmn = minimum([minimum(a), minimum(b)])
  
  function symlog(x, linthresh)
    sign_x = sign(x)
    abs_x = abs(x)
    scaled = linthresh * log10(abs_x / linthresh + 1)
    return sign_x * scaled
  end

  if UnicodePlotsPrint
    println(UnicodePlots.heatmap(symlog.(get_middle_nxn(a,75),0.0001), cmax = cmx, cmin = cmn, colormap=:inferno, title="Heatmap of star $starSample"))
    println(UnicodePlots.heatmap(symlog.(get_middle_nxn(b,75),0.0001), cmax = cmx, cmin = cmn, colormap=:inferno, title="Heatmap of Pixel Grid Fit $starSample"))
    println(UnicodePlots.heatmap(get_middle_nxn(a - b, 75), colormap=:inferno, title="Heatmap of Residuals"))
  end

  if cairomakiePlots
    testField(u, v) = Point2f(ds_du(u,v), ds_dv(u,v)) # x'(t) = -x, y'(t) = 2y
    u = range(minimum(u_coordinates), stop=maximum(u_coordinates), step=0.0001)            
    v = range(minimum(v_coordinates), stop=maximum(v_coordinates), step=0.0001)            

    s_map = [s(u,v) for u in u, v in v]
     
    fig1 = Figure(resolution = (1920, 1080), fontsize = 30, fonts = (;regular="CMU Serif"))
    ax1 = fig1[1, 1] = CairoMakie.Axis(fig1, xlabel = L"u", ylabel = L"v",xticklabelsize = 40, yticklabelsize = 40, xlabelsize = 40, ylabelsize = 40)
    fs1 = CairoMakie.heatmap!(ax1, u, v, s_map, colormap = Reverse(:plasma))
    CairoMakie.streamplot!(ax1,
                testField,
                u,
                v,
                colormap = Reverse(:plasma),
                gridsize = (32, 32),
                density = 0.25,
                arrow_size = 10)

    CairoMakie.Colorbar(fig1[1, 2],
                        fs1,
                        label = L"s(u,v)",
                        width = 20,
                        labelsize = 50,
                        ticklabelsize = 14)
     
    CairoMakie.colgap!(fig1.layout, 5)
     
    save("s_uv.png", fig1)

    testField(u, v) = Point2f(dg1_du(u,v), dg1_dv(u,v)) # x'(t) = -x, y'(t) = 2y
    u = range(minimum(u_coordinates), stop=maximum(u_coordinates), step=0.0001)            
    v = range(minimum(v_coordinates), stop=maximum(v_coordinates), step=0.0001)            

    g1_map = [g1(u,v) for u in u, v in v]
     
    fig2 = Figure(resolution = (1920, 1080), fontsize = 30, fonts = (;regular="CMU Serif"))
    ax2 = fig2[1, 1] = CairoMakie.Axis(fig2, xlabel = L"u", ylabel = L"v",xticklabelsize = 40, yticklabelsize = 40, xlabelsize = 40, ylabelsize = 40)
    fs2 = CairoMakie.heatmap!(ax2, u, v, g1_map, colormap = Reverse(:plasma))
    CairoMakie.streamplot!(ax2,
                testField,
                u,
                v,
                colormap = Reverse(:plasma),
                gridsize = (32, 32),
                density = 0.25,
                arrow_size = 10)

    CairoMakie.Colorbar(fig2[1, 2],
                        fs2,
                        label = L"g1(u,v)",
                        width = 20,
                        labelsize = 50,
                        ticklabelsize = 14)
     
    CairoMakie.colgap!(fig2.layout, 5)
     
    save("g1_uv.png", fig2)

    testField(u, v) = Point2f(dg2_du(u,v), dg2_dv(u,v)) # x'(t) = -x, y'(t) = 2y
    u = range(minimum(u_coordinates), stop=maximum(u_coordinates), step=0.0001)            
    v = range(minimum(v_coordinates), stop=maximum(v_coordinates), step=0.0001)            

    g2_map = [g2(u,v) for u in u, v in v]

    fig3 = Figure(resolution = (1920, 1080), fontsize = 30, fonts = (;regular="CMU Serif"))
    ax3 = fig3[1, 1] = CairoMakie.Axis(fig3, xlabel = L"u", ylabel = L"v", xticklabelsize = 40, yticklabelsize = 40, xlabelsize = 40, ylabelsize = 40)
    fs3 = CairoMakie.heatmap!(ax3, u, v, g2_map, colormap = Reverse(:plasma))
    CairoMakie.streamplot!(ax3,
                testField,
                u,
                v,
                colormap = Reverse(:plasma),
                gridsize = (32, 32),
                density = 0.25,
                arrow_size = 10)

    CairoMakie.Colorbar(fig3[1, 2],
                        fs3,
                        label = L"g2(u,v)",
                        width = 20,
                        labelsize = 50,
                        ticklabelsize = 14)
     
    CairoMakie.colgap!(fig3.layout, 5)
     
    save("g2_uv.png", fig3)
  end

  #=
  scale = 1/0.29
  ks93, k0 = ks(g1_map, g2_map)
  ksCosmos = get_middle_15x15(imfilter(ks93, Kernel.gaussian(scale)))
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

  Plots.savefig(kshm, joinpath("outdir","kaisserSquires.png"))
  =#
end

if UnicodePlotsPrint
  println(UnicodePlots.boxplot(["s model", "s data", "g1 model", "g1 data", "g2 model", "g2 data"], 
                               [s_model, s_data, g1_model, g1_data, g2_model, g2_data],
                              title="Boxplot of df.shopt"))
end

GC.gc()

# ---------------------------------------------------------#
fancyPrint("Saving Data to summary.shopt")
writeFitsData()

GC.gc()

# ---------------------------------------------------------#
fancyPrint("Done! =]")
end_time = Dates.now()
println("━ Total Time: ", (end_time - start) / Dates.Millisecond(1000), " seconds") 
#println("━ Total Time: ", Dates.format(now() - start, "HH:MM:SS"))
println("For more on ShOpt.jl, see https://github.com/EdwardBerman/shopt")
