using PyCall

function cataloging()
  py"""
  import numpy as np
  from astropy.io import fits
  from astropy.table import Table, hstack, vstack
  f = fits.open('/home/eddieberman/research/mcclearygroup/mosaic_nircam_f444w_COSMOS-Web_60mas_v0_1_starcat.fits')
  vignets = f[1].data['VIGNET']
  l = len(vignets)
  """
  v = py"vignets"
  catalog = py"list(map(np.array, $v))"
  r = size(catalog[1], 1)
  c = size(catalog[1], 2)
  return catalog, r, c, length(catalog)
end
