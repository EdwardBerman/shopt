function cataloging(args; nm=nanMask, nz=nanToZero)
  datadir = args[3]
  py"""
  import numpy as np
  from astropy.io import fits
  from astropy.table import Table, hstack, vstack
  python_datadir = $datadir
  f = fits.open(python_datadir)
  vignets = f[2].data['VIGNET']
  err_vignets = f[2].data['ERR_VIGNET']
  l = len(vignets)
  """
  datadir = py"python_datadir"
  v = py"vignets"
  err = py"err_vignets"
  catalog = py"list(map(np.array, $v))"
  errVignets = py"list(map(np.array, $err))"
  r = size(catalog[1], 1)
  c = size(catalog[1], 2)
  catalogNew = []
  for i in 1:length(catalog)
    #push!(catalogNew, catalog[i]./sum(catalog[i]))
    push!(catalogNew, nm(catalog[i])./sum(nz(nm(catalog[i]))))
    if i < 25
      a = nz(nm(catalog[i]))
    end
  end
  return catalogNew, errVignets, r, c, length(catalog)
end


