import numpy as np
import pdb
from astropy import units as u
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
f = fits.open('/home/eddieberman/research/mcclearygroup/mock_data/mosaics/mosaic_nircam_f115w_COSMOS-Web_i2d.fits')
w = WCS(f[1].header)

# Example xy coordinates and origin
xy_coords = np.array([[10.3, 20.4], [30.5, 40.6], [50.7, 60.8]])
origin = 1

# Call the function
result = w.all_pix2world(xy_coords, origin)
#result = w.pixel_to_world(xy_coords, 10000)
#ra = result.ra.deg  # Save RA as a float value
#dec = result.dec.deg  # Save Dec as a float value
    
#print("RA:", ra)
#print("Dec:", dec)
    

# Print the focal coordinates
print(result[1])



