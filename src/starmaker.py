import psfex
import galsim, galsim.des
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from astropy.table import Table
import ipdb

from .hsm_fitter import do_hsm_fit


class StarMaker():
    '''
    Class to store catalog stars and fit information.
     - Read in star entry from some catalog
     - Trim as needed, make a GS object. Get HSM fit.
     - This object will store all star vignets
    '''

    def __init__(self, star_cat, pix_scale, vb=False, vignet_size=None):
        '''
        star_cat is either SExtractor catalog,
        catalog file path, or simply list of np.ndarrays
        models are the PSF models: Galsim object or just np.ndarrays
        '''

        self.star_cat = star_cat
        self.pixel_scale = pix_scale
        self.vignet_size = vignet_size
        self.sky_level = 0.0
        self.sky_std = 0.0
        self.vb = vb

        self.x = []
        self.y = []
        self.models = []
        self.stamps = []
        self.star_flux = []
        self.star_mag = []
        self.err_stamps = []

        self.hsm_sig = []
        self.hsm_g1 = []
        self.hsm_g2 = []
        self.fwhm = []


    def _read_cat(self,vb=False):
        '''
        This could be expanded to do all the star catalog reading,
        but keep it simple for now & give it a pre-read star catalog
        '''

        #self.star_cat = star_cat

        if vb==True:
            print("fitting to %d stars"  % len(self.star_cat))

        return


    def _set_background(self,bg_obj=None,vb=False):
        '''
        bkg_obj is expected to be an instance of the
        StampBackground class
        '''
        if bg_obj is not None:
            self.sky_level = bg_obj.sky_level
            self.sky_std = bg_obj.sky_std

        else:
            self.sky_level = 0.0
            self.sky_std = 0.0

        if vb==True:
            print("sky level = %.3f +/- %.3f" % (self.sky_level, self.sky_std))

        return


    def _get_star_vignets(self, vb):
        '''
        Make star stamps from SExtractor catalog vignets. Also populate
        ERR stamps, which should have been added to star catalog.

        TO DO: add try/except to catch cases where the star catalog doesn't have
        an ERR column
        '''

        for i in range(len(self.star_cat)):

            this_vign = self.star_cat[i]['VIGNET']
            this_err_vign = self.star_cat[i]['ERR_VIGNET']
            x_pos = self.star_cat[i]['X_IMAGE'];
            y_pos = self.star_cat[i]['Y_IMAGE']
            star_mag = self.star_cat[i]['MAG_AUTO']
            star_flux = self.star_cat[i]['FLUX_AUTO']

            this_vign[this_vign <= -999] = np.nan
            this_vign[np.isnan(this_vign)] = self.sky_level
            vign_cutout = this_vign

            # Time to normalize stars
            star_sum = np.nansum(vign_cutout)
            vign_cutout = vign_cutout/star_sum

            if vb is True:
                print(f'Star {i} has flux {star_flux:.3f}')

            self.x.append(x_pos)
            self.y.append(y_pos)
            self.star_mag.append(star_mag)
            self.star_flux.append(star_flux)
            self.stamps.append(vign_cutout)
            self.models.append(vign_cutout)
            self.err_stamps.append(this_err_vign)

        self.x=np.array(self.x)
        self.y=np.array(self.y)

        # If a vignet_size wasn't supplied, set it to be the star VIGNET size
        if self.vignet_size == None:
            self.vignet_size = np.shape(this_vign)[1]
            print(f'Setting vignet size to {self.vignet_size}')
        return


    def run(self,vb=False,bg_obj=None):

        # Kind of does nothing right now but whatever
        self._read_cat(vb=vb)

        # Set backgrounds using supplied bkg_obj
        self._set_background(bg_obj=bg_obj,vb=vb)

        # Create star vignets
        self._get_star_vignets(vb=vb)

        # Create GS Object & record fit
        do_hsm_fit(self)

        return


class StampBackground():
    '''
    Determining and storing star cutout backgrounds for shape fitting purposes
    Maybe not efficient, but runs the first time on stars.

    : sky_level : median background of star stamp
    : sky_std   : standard deviation of star stamp
    : cat       : filepath, astropy.Table() instance or list of arrays
    : vclip     : side of sub-stamp to sample from later stamp in self.cat

    '''

    def __init__(self,cat=None,sky_level=None,sky_std=None):

        self.cat = cat
        self.vclip = 6
        self.sky_level = 0.0
        self.sky_std = 0.0
        self.substamps = []

        if sky_level is not None:
            print("setting sky level to %d" % sky_level)
            self.sky_level = sky_level
        if sky_std is not None:
            print("setting sky stdev to %d" % sky_std)
            self.sky_std = sky_std

    def _read_star_cat(self):
        pass

        return


    def calc_star_bkg(self, vb=False):
        '''
        Reading in file if needed, compute sky background of either
        SEXtractor VIGNETS or just arrays

        Is there better way to write this than a series of if statements?
        '''

        if self.cat is None:
            print("Catalog can't be 'None' if calculating sky background!")
            print("Please supply `cat` parameter")
            sys.exit()

        if type(self.cat) == 'str':
            obj_cat = Table.read(self.cat)
            cutouts = obj_cat['VIGNET']
        elif type(self.cat) == Table:
            cutouts=self.cat['VIGNET']
        elif (type(self.cat) == list) and (type(self.cat[0]) == np.ndarray):
            cutouts = self.cat
        else:
            # Go with God
            cutouts = self.cats

        self.sky_level, self.sky_std = self._calc_stamp_bkg(cutouts)

        if vb==True:
            print('star bkg = %.3f +/- %.3f' % (self.sky_level, self.sky_std))
        return self.sky_level, self.sky_std


    def _calc_stamp_bkg(self,cutouts):
        '''
        Input:
            cutouts : a list of np.ndarrays representing star images
        Output:
            sky_level : the median sky level in the star cutouts
            sky_std : standard deviation of sky level in star cutouts

        '''

        j = self.vclip

        for cutout in cutouts:
            cutout[cutout<= -999] = np.nan
            self.substamps.append(cutout[-j:,-j:])
            self.substamps.append(cutout[0:j,0:j])

        sky_level = np.nanmedian(self.substamps)
        sky_std = np.nanstd(self.substamps)

        return sky_level, sky_std
