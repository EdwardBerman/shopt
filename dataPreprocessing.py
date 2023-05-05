import webbpsf
nc = webbpsf.NIRCam()
psf = nc.calc_psf(nlambda=5, fov_arcsec=2)

