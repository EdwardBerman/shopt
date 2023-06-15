## .shopt file
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


function writeFitsData(meanRelativeError=meanRelativeError, s_model=s_model, g1_model=g1_model, g2_model=g2_model, s_data=s_data, g1_data=g1_data, g2_data=g2_data, u_coordinates = u_coordinates, v_coordinates = v_coordinates, PolynomialMatrix = PolynomialMatrix, outdir = outdir, configdir=configdir, starCatalog = starCatalog, pixelGridFits=pixelGridFits, errVignets=errVignets, fancyprint=fancyprint)
  
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

  hdu1 = fits.PrimaryHDU(PolynomialMatrix)
  hdu1.header['EXTNAME'] = 'Polynomial Matrix'
  
  c00 = fits.Column(name='u coordinates', array=u_coordinates, format='D')
  c01 = fits.Column(name='v coordinates', array=v_coordinates, format='D')
  c1 = fits.Column(name='s model', array=s_model, format='D')
  c2 = fits.Column(name='g1 model', array=g1_model, format='D')
  c3 = fits.Column(name='g2 model', array=g2_model, format='D')
  c4 = fits.Column(name='s data', array=s_data, format='D')
  c5 = fits.Column(name='g1 data', array=g1_data, format='D')
  c6 = fits.Column(name='g2 data', array=g2_data, format='D')
  c7 = fits.Column(name='mean relative error', array=meanRelativeError, format='D')

  VIGNETS_hdu = fits.ImageHDU(starCatalog)
  VIGNETS_hdu.header['EXTNAME'] = 'VIGNETS'

  errVignets_hdu = fits.ImageHDU(errVignets)
  errVignets_hdu.header['EXTNAME'] = 'Error Vignets'

  pixelGridFits_hdu = fits.ImageHDU(pixelGridFits)
  pixelGridFits_hdu.header['EXTNAME'] = 'Pixel Grid Fits'

  summary_statistics_hdu = fits.BinTableHDU.from_columns([c00, c01, c1, c2, c3, c4, c5, c6, c7]) 
  summary_statistics_hdu.header['EXTNAME'] = 'Summary Statistics'

  hdul = fits.HDUList([hdu1, summary_statistics_hdu, VIGNETS_hdu, errVignets_hdu, pixelGridFits_hdu])

  hdul.writeto('summary.shopt', overwrite=True)
  """

  command1 = `mv summary.shopt $outdir`
  run(command1)

  command2 = `cp $configdir/shopt.yml $outdir`
  run(command2)

  current_time = now()
  command3 = `mv $outdir/shopt.yml $outdir/$(Dates.format(Time(current_time), "HH:MM:SS")*"_shopt.yml")`
  run(command3)

  fancyprint("Producing Additional Python Plots")

  #command4 = `python $outdir/diagnostics.py`
  #run(command4)
  py"""
  from astropy.io import fits
  import numpy as np
  import matplotlib.pyplot as plt
  import matplotlib.colors as colors
  import matplotlib.cbook as cbook
  from matplotlib import cm
  from matplotlib import animation
  import imageio
  from mpl_toolkits.axes_grid1 import make_axes_locatable

  f = fits.open('/home/eddieberman/research/mcclearygroup/shopt/outdir/summary.shopt')

  polyMatrix = f[0].data


  def polynomial_interpolation_star(u,v, polynomialMatrix):
    r,c = np.shape(f[0].data)[0], np.shape(f[0].data)[1]
    star = np.zeros((r,c))
    for i in range(r):
      for j in range(c):
        star[i,j] = polynomialMatrix[i,j][0]*u**3 + \
          polynomialMatrix[i,j][1]*v**3 + \
          polynomialMatrix[i,j][2]*u**2*v + \
          polynomialMatrix[i,j][3]*u*v**2 + \
          polynomialMatrix[i,j][4]*u**2 + \
          polynomialMatrix[i,j][5]*v**2 + \
          polynomialMatrix[i,j][6]*u*v + \
          polynomialMatrix[i,j][7]*u + \
          polynomialMatrix[i,j][8]*v + \
          polynomialMatrix[i,j][9]
    star = star/np.sum(star)
    return star

  a = polynomial_interpolation_star(f[1].data['u coordinates'][0], f[1].data['v coordinates'][0]   ,polyMatrix)
  fig, axes = plt.subplots(1, 3)

  axes[0].imshow(f[4].data[0, :, :  ], norm=colors.SymLogNorm(linthresh=1*10**(-4)))
  axes[0].set_title('Pixel Grid Fit')

  # Display the second image in the second subplot
  axes[1].imshow(a, norm=colors.SymLogNorm(linthresh=1*10**(-4)))
  axes[1].set_title('Polynomial Interpolation')

  axes[2].imshow(f[4].data[3, :, :  ] - a, norm=colors.SymLogNorm(linthresh=1*10**(-4)))
  axes[2].set_title('Residuals')

  # Adjust the spacing between subplots
  plt.tight_layout()

  vignets = f[2].data
  a,b,c = np.shape(f[4].data)
  pixelGrid = np.zeros((b,c,a))

  for i in range(a):
      pixelGrid[:,:,i] = f[4].data[i,:,:]

  print(np.shape(pixelGrid))
  print(np.shape(vignets))

  fig2, axes2 = plt.subplots(1, 3)
  axes2[0].imshow(vignets[:,:,0], norm=colors.SymLogNorm(linthresh=1*10**(-4)))
  axes2[0].set_title('vignets')
  axes2[1].imshow(pixelGrid[:,:,0], norm=colors.SymLogNorm(linthresh=1*10**(-4)))
  axes2[1].set_title('pixel grid')
  axes2[2].imshow(vignets[:,:,0] - pixelGrid[:,:,0], norm=colors.SymLogNorm(linthresh=1*10**(-4)))
  axes2[2].set_title('Log Scale Residuals')
  plt.tight_layout()
  plt.show()


  def meanRelativeError(vignets, pixelGrid):
      meanRelativeError = np.zeros((vignets.shape[0], vignets.shape[1]))
      for j in range(vignets.shape[0]):
          for k in range(vignets.shape[1]):
              RelativeError = []
              for i in range(vignets.shape[2]):
                  RelativeError.append((vignets[j,k,i] - pixelGrid[j,k,i]) / (vignets[j,k,i]) + 1e-10)
              meanRelativeError[j,k] = np.mean(RelativeError)
      return meanRelativeError

  fig3, axes3 = plt.subplots(1)
  im = axes3.imshow(meanRelativeError(vignets, pixelGrid), cmap=plt.cm.bwr_r, norm=colors.SymLogNorm(linthresh=1*10**(-4)))
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
      pk.append(np.mean(radiusPixels))


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
  im = ax5.imshow(polyMatrix[:, :, 0], cmap='viridis', norm=colors.SymLogNorm(linthresh=1*10**(-4)), interpolation='nearest')

  frames = range(np.shape(polyMatrix)[2])  # Number of frames
  animation = animation.FuncAnimation(fig5, update, frames=frames, interval=200, blit=True)

  fig.savefig('pgfVsInterp.png')
  fig2.savefig('logV_P_R.png')
  fig3.savefig('MRE.png')
  fig4.savefig('fftPk.png')


  """

  # run on sampled indices, copy diagnostics.py to py""" """ here
end
