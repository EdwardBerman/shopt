import numpy as np
from astropy.table import Table, vstack
import pdb
import glob
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from src.utils import read_yaml
import fitsio
import astropy.nddata as nd

class BoxCutter:
    def __init__(self, config_file, image_file=None,
                        box_size=None, x=None, y=None):
        '''
        Busted cookie-cutter
        '''
        self.config_file = config_file
        self.image_file = image_file
        self.x = x
        self.y = y
        self.image = []
        self.error = []

        self.config = read_yaml(self.config_file)

        # We refer to box size a lot so read it in, w/e
        self.box_size = np.int(self.config['box_size'])


    def _grab_wcs_box(self, obj):
        '''
        Placeholder for what might be a good RA/Dec matcher
        '''
        ra_tag = self.config['input_catalog']['ra_tag']
        dec_tag = self.config['input_catalog']['dec_tag']
        ra_unit = self.config['input_catalog']['ra_unit']
        dec_unit = self.config['input_catalog']['dec_unit']

        coord = SkyCoord(ra=obj[ra_tag]*ra_unit,
                            dec=obj[dec_tag]*dec_unit
                            )
        #c = SkyCoord([1, 2, 3], [-30, 45, 8], frame="icrs", unit="deg")
        wcs = WCS(fits.getheader(self.image_file))

        x, y = wcs.all_world2pix(coord.ra.value, coord.dec.value, 0)

        object_pos_in_image = [x.item(), y.item()]

        return


    def _grab_box(self, x, y):
        '''
        WHAT'S IN THE BAAAHHX:
            im: should be array-like format
            x, y: location of star from catalog
            box_size: vignet size to cut around star
        Returns:
            WHAT'S IN THE BAAAAHX
        '''

        bs = np.int(self.box_size)
        bb = self.box_size/2
        im = self.image
        j1 = int(np.floor(x-bb))
        j2 = int(np.floor(x+bb))
        k1 = int(np.floor(y-bb))
        k2 = int(np.floor(y+bb))

        '''
        try:
            box = im[k1:k2, j1:j2]
        except:
            pdb.set_trace()
        if np.shape(box) != (bs, bs):
            box = np.zeros([bs, bs])
        '''

        # No one told me I could just use this
        box = nd.Cutout2D(data=im, position=(x,y),
                    size=self.box_size, copy=True, mode='partial')

        return box.data


    def grab_boxes(self, image_file, cat_file):
        '''
        Load image files, call box grabber
        '''
        config = self.config
        cat_hdu = config['input_catalog']['hdu']
        x_tag = config['input_catalog']['x_tag']
        y_tag = config['input_catalog']['y_tag']

        hdu = config['err_image']['hdu']
        box_size = config['box_size']

        if type(image_file) is str:
            imf = fitsio.FITS(image_file, 'r')[hdu]
            self.image = imf.read()
        else:
            self.image = image_file

        if type(cat_file) is str:
            starcat_fits = fitsio.FITS(cat_file, 'rw')
            starcat = starcat_fits[cat_hdu].read()
        else:
            starcat = cat_file

        x = starcat[x_tag]; y = starcat[y_tag]
        new_boxes = list(map(self._grab_box, x, y))

        return new_boxes
