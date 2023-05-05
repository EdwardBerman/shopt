using PyCall
fits = pyimport("astropy.io.fits")
Table = pyimport("astropy.table")
f = fits.open("data/DR16Q_v4.fits")
