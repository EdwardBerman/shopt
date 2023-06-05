config = YAML.load_file(joinpath(configdir, "shopt.yml"))

function signal_to_noise(I, e; nm=nanMask, nz=nanToZero)
  signal_power = sum(nz(nm(I)).^2)
  noise_power = sum(e.^2)
  snr = 10*log10(signal_power/noise_power)
  return snr
end

function cataloging(args; nm=nanMask, nz=nanToZero, snr=signal_to_noise, dout=outliers_filter)
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
  signal2noiseRatios = []
  for i in 1:length(catalog)
    #push!(catalogNew, catalog[i]./sum(catalog[i]))
    push!(catalogNew, nm(catalog[i])./sum(nz(nm(catalog[i]))))
    push!(signal2noiseRatios, snr(catalog[i], errVignets[i]))
  end
  catalogNew, errVignets = dout(signal2noiseRatios, catalogNew, errVignets)
  println("Removed $(length(catalog) - length(catalogNew)) outliers on the basis of Signal to Noise Ratio")
  return catalogNew, errVignets, r, c, length(catalogNew)
end


