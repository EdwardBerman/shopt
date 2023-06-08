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


function writeFitsData(s_model=s_model, g1_model=g1_model, g2_model=g2_model, s_data=s_data, g1_data=g1_data, g2_data=g2_data, u_coordinates = u_coordinates, v_coordinates = v_coordinates, PolynomialMatrix = PolynomialMatrix, outdir = outdir, starCatalog = starCatalog, pixelGridFits=pixelGridFits, errVignets=errVignets)
  
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

  VIGNETS_hdu = fits.ImageHDU(starCatalog)
  VIGNETS_hdu.header['EXTNAME'] = 'VIGNETS'

  errVignets_hdu = fits.ImageHDU(errVignets)
  errVignets_hdu.header['EXTNAME'] = 'Error Vignets'

  pixelGridFits_hdu = fits.ImageHDU(pixelGridFits)
  pixelGridFits_hdu.header['EXTNAME'] = 'Pixel Grid Fits'

  summary_statistics_hdu = fits.BinTableHDU.from_columns([c00, c01, c1, c2, c3, c4, c5, c6]) 
  summary_statistics_hdu.header['EXTNAME'] = 'Summary Statistics'

  hdul = fits.HDUList([hdu1, summary_statistics_hdu, VIGNETS_hdu, errVignets_hdu, pixelGridFits_hdu])

  hdul.writeto('summary.shopt', overwrite=True)
  """

  command = `mv summary.shopt $outdir`
  run(command)


end
