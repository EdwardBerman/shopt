## .shopt file
using PyCall
using Dates

#=
Functions to be called to write output to a fits file and add more diagnostic plots
=#

function writeData(size, shear1, shear2, sizeD, shear1D, shear2D)
  df = DataFrame(star = 1:length(size), 
                 s_model=size, 
                 g1_model=shear1, 
                 g2_model=shear2, 
                 s_data=sizeD, 
                 g1_data=shear1D, 
                 g2_data=shear2D)

  CSV.write(joinpath("outdir", "df.shopt"), df)
end


function readData()
  DataFrame(CSV.File(joinpath("outdir", "df.shopt")))
end

function writeFitsData(sampled_indices=sampled_indices, meanRelativeError=meanRelativeError, s_model=s_model, g1_model=g1_model, g2_model=g2_model, s_data=s_data, g1_data=g1_data, g2_data=g2_data, u_coordinates = u_coordinates, v_coordinates = v_coordinates, PolynomialMatrix = PolynomialMatrix, outdir = outdir, configdir=configdir, starCatalog = starCatalog, pixelGridFits=pixelGridFits, errVignets=errVignets, fancyPrint=fancyPrint, training_stars=training_stars, training_u_coords=training_u_coords, training_v_coords=training_v_coords, validation_stars=validation_stars, validation_u_coords=validation_u_coords, validation_v_coords=validation_v_coords, validation_star_catalog=validation_star_catalog, degree=degree, YAMLSAVE=YAMLSAVE, parametersHistogram=parametersHistogram, parametersScatterplot=parametersScatterplot, cairomakiePlots=cairomakiePlots, pythonPlots=pythonPlots)
  
  m, n = size(starCatalog[1])
  array_3d = zeros(m, n, length(starCatalog))
  for (i, array) in enumerate(starCatalog)
    array_3d[:, :, i] = array
  end
  starCatalog = array_3d
  
  m, n = size(pixelGridFits[1])
  array_3d = zeros(m, n, length(pixelGridFits))
  for (i, array) in enumerate(pixelGridFits)
    array_3d[:, :, i] = array
  end
  pixelgridfits = array_3d

  m, n = size(errVignets[1])
  array_3d = zeros(m, n, length(errVignets))
  for (i, array) in enumerate(errVignets)
    array_3d[:, :, i] = array
  end
  errVignets = array_3d

  py"""
  from astropy.io import fits
  import numpy as np
  
  s_model = np.array($s_model, dtype=np.float64)
  g1_model = np.array($g1_model, dtype=np.float64)
  g2_model = np.array($g2_model, dtype=np.float64)
  s_data = np.array($s_data, dtype=np.float64)
  g1_data = np.array($g1_data, dtype=np.float64)
  g2_data = np.array($g2_data, dtype=np.float64)
  u_coordinates = np.array($u_coordinates, dtype=np.float64)
  v_coordinates = np.array($v_coordinates, dtype=np.float64)
  PolynomialMatrix = np.array($PolynomialMatrix, dtype=np.float64)
  starCatalog = np.array($starCatalog, dtype=np.float64)
  pixelGridFits = np.array($pixelGridFits, dtype=np.float64)
  errVignets = np.array($errVignets, dtype=np.float64)
  meanRelativeError = np.array($meanRelativeError, dtype=np.float64)
  training_u_coords = np.array($training_u_coords, dtype=np.float64)
  training_v_coords = np.array($training_v_coords, dtype=np.float64)
  validation_u_coords = np.array($validation_u_coords, dtype=np.float64)
  validation_v_coords = np.array($validation_v_coords, dtype=np.float64)
  training_stars = np.array($training_stars, dtype=np.float64)
  validation_stars = np.array($validation_stars, dtype=np.float64)
  deg_element = $degree
  degree_array = np.array([deg_element], dtype=np.float64)

  hdu1 = fits.PrimaryHDU(PolynomialMatrix)
  hdu1.header['EXTNAME'] = 'Polynomial Matrix'
  
  c00 = fits.Column(name='u coordinates', array=u_coordinates, format='D')
  c01 = fits.Column(name='v coordinates', array=v_coordinates, format='D')
  c02 = fits.Column(name='Training_u_coords', array=training_u_coords, format='D')
  c03 = fits.Column(name='Training_v_coords', array=training_v_coords, format='D')
  c04 = fits.Column(name='validation_u_coords', array=validation_u_coords, format='D')
  c05 = fits.Column(name='validation_v_coords', array=validation_v_coords, format='D')
  c1 = fits.Column(name='s model', array=s_model, format='D')
  c2 = fits.Column(name='g1 model', array=g1_model, format='D')
  c3 = fits.Column(name='g2 model', array=g2_model, format='D')
  c4 = fits.Column(name='s data', array=s_data, format='D')
  c5 = fits.Column(name='g1 data', array=g1_data, format='D')
  c6 = fits.Column(name='g2 data', array=g2_data, format='D')
  c7 = fits.Column(name='mean relative error', array=meanRelativeError, format='D')
  c8 = fits.Column(name='polynomial degree', array=degree_array, format='D')

  VIGNETS_hdu = fits.ImageHDU(starCatalog)
  VIGNETS_hdu.header['EXTNAME'] = 'VIGNETS'

  errVignets_hdu = fits.ImageHDU(errVignets)
  errVignets_hdu.header['EXTNAME'] = 'Error Vignets'

  pixelGridFits_hdu = fits.ImageHDU(pixelGridFits)
  pixelGridFits_hdu.header['EXTNAME'] = 'Pixel Grid Fits'

  validation_hdu = fits.ImageHDU(validation_stars)
  validation_hdu.header['EXTNAME'] = 'Validation Stars'

  summary_statistics_hdu = fits.BinTableHDU.from_columns([c00, c01, c02, c03, c04, c05, c1, c2, c3, c4, c5, c6, c7, c8]) 
  summary_statistics_hdu.header['EXTNAME'] = 'Summary Statistics'

  hdul = fits.HDUList([hdu1, summary_statistics_hdu, VIGNETS_hdu, errVignets_hdu, pixelGridFits_hdu, validation_hdu])

  hdul.writeto('summary.shopt', overwrite=True)
  """
  command1 = `mv summary.shopt $outdir`
  run(command1)

  if YAMLSAVE
    command2 = `cp $configdir/shopt.yml $outdir`
    run(command2)

    current_time = now()
    command3 = `mv $outdir/shopt.yml $outdir/$(Dates.format(Time(current_time), "HH:MM:SS")*"_shopt.yml")`
    run(command3)
  end

  #command4 = `python $outdir/diagnostics.py`
  #run(command4)
  py_outdir = outdir*"/$(now())"
  if pythonPlots
    fancyPrint("Producing Additional Python Plots")
  else
    py"""
    import os 
    py_outdir = $py_outdir
    
    if not os.path.exists(py_outdir):
      os.makedirs(py_outdir)

    """
  end
  println("Creating ", py_outdir, " directory...")
  
  deg = degree 

  if pythonPlots
    py"""
    from astropy.io import fits
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    import matplotlib.cbook as cbook
    from matplotlib import cm
    from matplotlib import animation
    import imageio
    import os
    from mpl_toolkits.axes_grid1 import make_axes_locatable


    plt.ion()
    
    #Generalize this
    python_current_outdir = $outdir

    f = fits.open(python_current_outdir + '/summary.shopt')
    #f = fits.open('/home/eddieberman/research/mcclearygroup/shopt/outdir/summary.shopt')

    polyMatrix = f[0].data
    #print(polyMatrix[30,30,:])
    deg = $deg
    def polynomial_interpolation_star(u,v, polynomialMatrix):
        r,c = np.shape(f[0].data)[0], np.shape(f[0].data)[1]
        star = np.zeros((r,c))
        for i in range(r):
            for j in range(c):
              def objective_function(p, x, y, degree):
                num_coefficients = (degree + 1) * (degree + 2) // 2
                value = 0
                counter = 0
                for a in range(1, degree + 2):
                  for b in range(1, degree + 2):
                    if (a - 1) + (b - 1) <= degree:
                      value += p[counter] * x**(a - 1) * y**(b - 1)  
                      counter += 1
                return value 
              star[i,j] = objective_function(polynomialMatrix[i,j], u, v, deg)

        star = star/np.sum(star)
        return star

    python_sampled_indices = $sampled_indices
    sample_one = python_sampled_indices[0]
    sample_two = python_sampled_indices[1]
    sample_three = python_sampled_indices[2]

    a = polynomial_interpolation_star(f[1].data['validation_u_coords'][sample_one], f[1].data['validation_v_coords'][sample_one]   ,polyMatrix)
    #print(np.shape(a))
    fig, axes = plt.subplots(2, 3)

    # Display the first image in the first subplot
    axes[0, 0].imshow(f[5].data[sample_one, :, :  ], norm=colors.SymLogNorm(linthresh=1*10**(-6)))
    axes[0, 0].set_title('Pixel Grid Fit')

    # Display the second image in the second subplot
    axes[0, 1].imshow(a, norm=colors.SymLogNorm(linthresh=1*10**(-6)))
    axes[0, 1].set_title('Polynomial Interpolation')

    axes[0, 2].imshow(f[5].data[sample_one, :, :  ] - a, norm=colors.SymLogNorm(linthresh=1*10**(-6)))
    axes[0, 2].set_title('Residuals')
    
    b = polynomial_interpolation_star(f[1].data['validation_u_coords'][sample_two], f[1].data['validation_v_coords'][sample_two], polyMatrix)

    axes[1, 0].imshow(f[5].data[sample_two, :, :  ], norm=colors.SymLogNorm(linthresh=1*10**(-6)))
    axes[1, 0].set_title('Pixel Grid Fit')

    # Display the second image in the second subplot
    axes[1, 1].imshow(b, norm=colors.SymLogNorm(linthresh=1*10**(-6)))
    axes[1, 1].set_title('Polynomial Interpolation')

    axes[1, 2].imshow(f[5].data[sample_two, :, :  ] - b, norm=colors.SymLogNorm(linthresh=1*10**(-6)))
    axes[1, 2].set_title('Residuals')
    # Adjust the spacing between subplots
    plt.tight_layout()
    plt.show()



    vignets = f[2].data
    a,b,c = np.shape(f[4].data)
    pixelGrid = np.zeros((b,c,a))

    for i in range(a):
        pixelGrid[:,:,i] = f[4].data[i,:,:]

    #print(np.shape(pixelGrid))
    #print(np.shape(vignets))

    fig2, axes2 = plt.subplots(1, 3)
    axes2[0].imshow(vignets[:,:,0], norm=colors.SymLogNorm(linthresh=1*10**(-6)))
    axes2[0].set_title('vignets')
    axes2[1].imshow(pixelGrid[:,:,0], norm=colors.SymLogNorm(linthresh=1*10**(-6)))
    axes2[1].set_title('pixel grid')
    axes2[2].imshow(vignets[:,:,0] - pixelGrid[:,:,0], norm=colors.SymLogNorm(linthresh=1*10**(-6)))
    axes2[2].set_title('Log Scale Residuals')
    plt.tight_layout()
    plt.show()


    def meanRelativeError(vignets, pixelGrid):
        meanRelativeError = np.zeros((vignets.shape[0], vignets.shape[1]))
        for j in range(vignets.shape[0]):
            for k in range(vignets.shape[1]):
                RelativeError = []
                for i in range(vignets.shape[2]):
                  RelativeError.append((vignets[j,k,i] - pixelGrid[j,k,i]) / (vignets[j,k,i] + 1e-10))
                  meanRelativeError[j,k] = np.nanmean(RelativeError)
        return meanRelativeError

    fig3, axes3 = plt.subplots(1)
    im = axes3.imshow(meanRelativeError(vignets, pixelGrid), cmap=plt.cm.bwr_r, norm=colors.SymLogNorm(linthresh=1*10**(-6)))
    axes3.set_title('Mean Relative Error')
    divider = make_axes_locatable(axes3)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig3.colorbar(im, cax=cax)

    fft_image = np.fft.fft(vignets[:,:,0] - pixelGrid[:,:,0])
    fft_image = np.abs(fft_image) ** 2

    pk = []
    for i in range(1, 11):
        radius = np.linspace(1, max(vignets.shape[0]/2, vignets.shape[1]/2) - 1, num=10)
        radiusPixels = []
        for u in range(fft_image.shape[0]):
            for v in range(fft_image.shape[1]):
                if round(np.sqrt((u - fft_image.shape[0]/2)**2 + (v - fft_image.shape[1]/2)**2) - radius[i-1]) == 0:
                    radiusPixels.append(fft_image[u, v])
        pk.append(np.nanmean(radiusPixels))


    fig4, axes4 = plt.subplots(1,2)
    axes4[0].imshow(fft_image)
    axes4[0].set_title('FFT of Residuals')
    axes4[1].plot(radius, pk)
    axes4[1].set_title('Power Spectra')

    def update(frame):
        im.set_array(polyMatrix[:, :, frame]) # Update the plot data for each frame
        return im,

    fig5 = plt.figure()
    ax5 = fig5.add_subplot(111)
    im = ax5.imshow(polyMatrix[:, :, 0], cmap=plt.cm.bwr_r, norm=colors.SymLogNorm(linthresh=1*10**(-6)), interpolation='nearest')

    frames = range(np.shape(polyMatrix)[2])  # Number of frames
    animation = animation.FuncAnimation(fig5, update, frames=frames, interval=200, blit=True)

    def mreHist(vignets, pixelGrid):
        RelativeError = []
        for j in range(vignets.shape[0]):
            for k in range(vignets.shape[1]):
                for i in range(vignets.shape[2]):
                    RelativeError.append((vignets[j,k,i] - pixelGrid[j,k,i]) / (vignets[j,k,i]) + 1e-10)
        return RelativeError

    fig6, axes6 = plt.subplots(1, figsize=(30, 10))
    bins = np.linspace(-1, 1, 21)
    axes6.hist(mreHist(vignets, pixelGrid), bins = bins, color = "lightblue", ec="red", lw=3) #bins = bins
    axes6.set_xlabel('Relative Error', fontsize=20)
    axes6.set_ylabel('Frequency',fontsize=20)
    axes6.set_yscale('log')
    axes6.set_title(f'Relative Error Histogram (logscale)\n $\mu = {np.nanmean(mreHist(vignets, pixelGrid))}$,  $\sigma = {np.nanstd(mreHist(vignets, pixelGrid))/np.sqrt(len(mreHist(vignets, pixelGrid)) - np.sum(np.isnan(mreHist(vignets, pixelGrid)))   )}$', fontsize=30)

    py_outdir = $py_outdir
    
    if not os.path.exists(py_outdir):
      os.makedirs(py_outdir)

    fig.savefig(os.path.join(py_outdir,'pgfVsInterp.png'))
    fig2.savefig(os.path.join(py_outdir, 'logV_P_R.png'))
    fig3.savefig(os.path.join(py_outdir, 'MRE.png'))
    fig4.savefig(os.path.join(py_outdir, 'fftPk.png'))
    fig6.savefig(os.path.join(py_outdir, 'RelativeErrorHistogram.png'))
    #fig6.savefig(os.path.join(py_outdir, 'RelativeErrorHistogram.pdf'))

    """
  end 

  command4 = `mv $outdir/$(Dates.format(Time(current_time), "HH:MM:SS")*"_shopt.yml")  $py_outdir`
  run(command4)
  
  command5 = `mv $outdir/summary.shopt  $py_outdir`
  run(command5)
  
  if cairomakiePlots 
    command6 = `mv s_uv.png  $py_outdir`
    run(command6)
    
    command7 = `mv g1_uv.png  $py_outdir`
    run(command7)
    
    command8 = `mv g2_uv.png  $py_outdir`
    run(command8)
  end

  if parametersScatterplot
    command9 = `mv parametersScatterplot.png  $py_outdir`
    run(command9)

    command10 = `mv parametersScatterplot.pdf  $py_outdir`
    run(command10)
  end
  
  if parametersHistogram
    command11 = `mv parametersHistogram.pdf  $py_outdir`
    run(command11)

    command12 = `mv parametersHistogram.png  $py_outdir`
    run(command12)
  end

  # run on sampled indices, copy diagnostics.py to py""" """ here
end

# function P(u,v)
# PSF at uv
# end
