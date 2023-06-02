
import psfex
import galsim,galsim.des
import piff
from matplotlib import rc,rcParams
rc('font',**{'family':'serif'})
rc('text', usetex=True)

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from astropy.table import Table
import pdb, ipdb
import treecorr
import glob

from .starmaker import StarMaker, StampBackground
from .hsm_fitter import do_hsm_fit
from .plotter import make_resid_plot, plot_rho_stats
from .quiverplot import QuiverPlot
from .residplots import ResidPlots

class PSFMaker:
    '''
    Do HSM fits to psf vignettes
    Save output to file
    Could even make a few correlation functions if you like

    Note: psf_type is
    Possible improvements: make more general?
    '''

    def __init__(self, psf_file, pix_scale, vignet_size=None, psf_type='piff',
                        noisefree=False, rho_params=None, vb=False):
        '''
        psf_obj is the file name of psf,
        or alternatively an instance of it

        psf_type is the method to use to render the PSF.
        Should be one of the following:
            - 'epsfex:' Erin Sheldon psfex Python package
            - 'gpsfex:' Use galsim.des PSFEx method
            - 'piff:  ' Use PIFF rendering
        '''

        self.psf = psf_file
        self.psf_type = psf_type
        self.noisefree = noisefree
        self.vignet_size = vignet_size
        self.pixel_scale = pix_scale
        self.rho_params = rho_params
        self.vb = vb

        self.models = []
        self.stamps = []
        self.resids = []

        self.sky_level = 0.0
        self.sky_std = 0.0
        self.sample_scale = 1.0

        self.x = []
        self.y = []
        self.hsm_sig = []
        self.hsm_g1 = []
        self.hsm_g2 = []
        self.fwhm = []

        if psf_type not in ['epsfex','gpsfex','piff']:
            print("psf_type is the method to use to render the PSF,")
            print("Should be one of the following:")
            print("     - 'epsfex': Erin Sheldon psfex Python package")
            print("     - 'gpsfex': Use galsim.des PSFEx method")
            print("     - 'piff'  : Use PIFF rendering")


    def render_psf(self,x=None,y=None,flux=None,psf_type='epsfex'):
        '''
        Method to decide which rendering method to call, then call it.
        Appends PSF cutouts to self

        would be cool to try to make an assertion error pop out
        '''

        vb = self.vb

        if self.psf_type=='epsfex':
            if vb==True: print("rendering epsfex psf")
            self.sample_scale = self.psf['psf_samp']
            psf_im = self._make_pexim(x_pos=x, y_pos=y, flux=flux, vb=vb)

        elif self.psf_type=='gpsfex':
            if vb==True: print("rendering psfex_des psf")
            self.sample_scale = self.psf.sample_scale
            psf_im = self._make_gpsf(x_pos=x, y_pos=y, flux=flux, vb=vb)

        elif self.psf_type=='piff':
            if vb==True: print("rendering PIFF psf")
            #self.sample_scale = self.psf.single_psf.model.scale
            self.sample_scale = 1
            psf_im = self._make_piff(x_pos=x, y_pos=y, flux=flux, vb=vb)

        else:
            allowed=['piff','epsfex','gpsfex']
            print("PSF not one of ['piff','epsfex','gpsfex']")

        return psf_im

    def _make_pexim(self,x_pos,y_pos,flux=None,vb=False):
        '''
        Generate a esheldon.psfex rendering at position x_pos,y_pos
        incorporating sky noise & star flux
        '''

        pix_scale = self.pixel_scale
        im_wcs = galsim.PixelScale(self.pixel_scale)

        if flux == None:
            print("using unit flux")
            star_flux = 1
        else:
            star_flux=flux
            if vb==True: print("using flux=%.2f" % flux)

        this_pexim = self.psf.get_rec(y_pos,x_pos)

        # You had better hope this is odd
        '''
        # This could eventually be a call to BoxCutter...
        if this_pexim.shape[0] != self.vignet_size:
            sm = min(this_pexim.shape[0], self.vignet_size)
            lg = max(this_pexim.shape[0], self.vignet_size)
            n = int((lg - sm)/2)
            this_pexim = this_pexim[n:-n,n:-n]
        '''

        if this_pexim.shape[0] != self.vignet_size:
            n = int((this_pexim.shape[0]-self.vignet_size)/2)
            this_pexim = this_pexim[n:-n,n:-n]


        if self.noisefree == False:
            # Original was self.sky_std
            if vb == True: print("adding noise")
            noise = np.random.normal(loc=0,
                        scale=1e-5,size=this_pexim.shape
                        )
            this_pexim+=noise

        # Now we normalize
        pexim_rs = this_pexim #/np.sum(this_pexim)

        return pexim_rs


    def _make_gpsf(self,x_pos,y_pos,flux=None,vb=False):
        '''
        Generate a gs.des.des_psfex() image at position x_pos,y_pos
        incorporating sky noise & star flux (if desired).
        NOTE: you do need to set scale=pix_scale if image is oversampled
        '''

        pix_scale = self.pixel_scale
        im_wcs = galsim.PixelScale(pix_scale)

        if flux == None:
            star_flux = 1
        else:
            star_flux=flux
            if vb==True:print("using flux=%.2f" % flux)

        psfex_des = self.psf

        this_pos = galsim.PositionD(x_pos,y_pos)
        this_psf_des = psfex_des.getPSF(this_pos)
        gpsf_im = this_psf_des.drawImage(method='no_pixel',
                    nx=self.vignet_size, ny=self.vignet_size, scale=pix_scale)

        if self.noisefree==False:
            if vb == True: print("adding noise")
            sky_noise = galsim.GaussianNoise(sigma=1e-5)
            gpsf_im.addNoise(sky_noise)

        # Now we normalize to one -- not with galsim objects!
        gpsf_im_rs = gpsf_im #/np.sum(gpsf_im.array)

        return gpsf_im_rs

    def _make_piff(self, x_pos, y_pos, flux=None, vb=False):
        '''
        Render a PIFF psf model and add noise
        piff.PSF.draw returns a GalSim image of PSF, so can treat
        it in the same way as gpsf/des_psfex
        docs:
        https://rmjarvis.github.io/Piff/_build/html/psf.html?highlight=draw#

        Note piff.draw() outputs complex WCS and so calculateFWHM() will fail
        unless array part is redrawn into a gs.Image() with a pixel scale WCS.
        '''

        if flux == None:
            star_flux = 1
        else:
            star_flux=flux
            if vb == True: print("using flux=%.2f" % flux)

        piff_psf = self.psf
        piff_im = piff_psf.draw(x=x_pos,y=y_pos,
                    stamp_size=self.vignet_size)

        if self.noisefree==False:
            sky_noise = galsim.GaussianNoise(sigma=1e-5)
            piff_im.addNoise(sky_noise)
            if vb == True: print("Noise added")

        # Now we normalize to one -- not with galsim images!
        piff_im_rs = piff_im #/np.sum(piff_im)

        return piff_im_rs


    def run_rho_stats(self, stars, rho_params, vb=False, outdir='./'):
        '''
        Method to obtain rho-statistics for current PSF fit & save plots
        Requires StarMaker to be provided through 'stars' parameter

        default rho_params={'min_sep':100,'max_sep':3000,'nbins':60}
        '''

        # First do calculations
        rho1,rho2,rho3,rho4,rho5 = self._run_rho_stats(stars,
            rho_params=rho_params, vb=vb, outdir=outdir)

        # Then make plots
        outname=os.path.join(outdir,'_'.join([str(self.psf_type),'rho_stats']))
        plot_rho_stats(rho1, rho2, rho3, rho4, rho5,
                        pixel_scale=self.pixel_scale,
                        outname=outname
                        )

        print("Finished rho stat computation & plotting")

        return


    def _run_rho_stats(self, stars, rho_params, outdir=None, vb=False):

        min_sep = rho_params['min_sep']
        max_sep = rho_params['max_sep']
        nbins = rho_params['nbins']

        # Define quantities to be used
        wg = (self.hsm_g1 > -9999) & (stars.hsm_g1 > -9999)
        star_g1 = stars.hsm_g1[wg]
        star_g2 = stars.hsm_g2[wg]
        psf_g1 = self.hsm_g1[wg]
        psf_g2 = self.hsm_g2[wg]

        dg1 = star_g1 - psf_g1
        dg2 = star_g2 - psf_g2

        T  = 2.0*(self.hsm_sig[wg]**2)
        Tpsf = 2.0*(stars.hsm_sig[wg]**2)
        dT = T-Tpsf; dTT = dT/T

        # Stars & size-residual-scaled stars
        starcat = treecorr.Catalog(x=self.x[wg],y=self.y[wg],g1=star_g1,g2=star_g2)
        rs_starcat = treecorr.Catalog(x=self.x[wg],y=self.y[wg],g1=star_g1*dTT,g2=star_g2*dTT)

        # PSFs & size-residual-scaled PSFs
        psfcat = treecorr.Catalog(x=self.x[wg],y=self.y[wg],g1=psf_g1,g2=psf_g2)
        rs_psfcat = treecorr.Catalog(x=self.x[wg],y=self.y[wg],g1=psf_g1*dTT,g2=psf_g2*dTT)

        # PSF Resids
        psf_resid_cat = treecorr.Catalog(x=self.x[wg],y=self.y[wg],g1=dg1,g2=dg2)

        # rho-1: psf_ellip residual autocorrelation
        rho1 = treecorr.GGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins)
        rho1.process(psf_resid_cat)
        rho1.write(os.path.join(outdir,'_'.join([str(self.psf_type),'rho_1.txt'])))
        if vb==True:
            print('bin_size = %.6f' % rho1.bin_size)
            print('mean rho1 = %.4e median = %.4e std = %.4e' %
                (np.mean(rho1.xip),np.median(rho1.xip),
                np.std(rho1.xip)))

        # rho-2: psf_ellip x psf_ellip residual correlation
        rho2 = treecorr.GGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins)
        rho2.process(starcat,psf_resid_cat)
        rho2.write(os.path.join(outdir,'_'.join([str(self.psf_type),'rho_2.txt'])))
        if vb==True:
            print('mean rho2 = %.4e median = %.4e std = %.4e' %
                (np.mean(rho2.xip),np.median(rho2.xip),
                np.std(rho2.xip)))

        # My *guess* at rho-3: psf_ellip x vignet_size residual correlation
        rho3 = treecorr.GGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins)
        rho3.process(rs_starcat)
        rho3.write(os.path.join(outdir,'_'.join([str(self.psf_type),'rho_3.txt'])))
        if vb==True:
            print('mean rho3 = %.4e median = %.4e std = %.4e' %
                (np.mean(rho3.xip),np.median(rho3.xip),
                np.std(rho3.xip)))

        # My *guess* at rho-4: psf ellip resid x (psf ellip *size resid)
        rho4 = treecorr.GGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins)
        rho4.process(psf_resid_cat,rs_starcat)
        rho4.write(os.path.join(outdir,'_'.join([str(self.psf_type),'rho_4.txt'])))
        if vb==True:
            print('mean rho4 = %.4e median = %.4e std = %.4e' %
            (np.mean(rho4.xip),np.median(rho4.xip), np.std(rho4.xip)))

        # My *guess* at rho-4: psf ellip  x (psf ellip *size resid)
        rho5 = treecorr.GGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins)
        rho5.process(starcat,rs_starcat)
        rho5.write(os.path.join(outdir,'_'.join([str(self.psf_type),'rho_5.txt'])))

        if vb==True:
            print('mean rho5 = %.4e median = %.4e std = %.4e' %
            (np.mean(rho5.xip),np.median(rho5.xip), np.std(rho5.xip)))

        return rho1, rho2, rho3, rho4, rho5


    def run_all(self, stars, vb=False, outdir='./psf_diagnostics'):
        '''
        stars is expected to be an instance of the StarMaker class
        Possible improvements: allow user to supply just X,Y?
        Allow a freestanding bg value?
        '''

        if type(stars) is not StarMaker:
            print("StarMaker() instance not supplied, exiting")
            sys.exit()

        self.sky_level = stars.sky_level
        self.sky_std = stars.sky_std
        self.x = stars.x
        self.y = stars.y

        if self.vignet_size == None:
            self.vignet_size = stars.vignet_size
            print(f'PSFmaker vignet size is {self.vignet_size}')

        # Render PSF, take residual against stars
        for i in range(len(stars.x)):
            xpos = stars.x[i]; ypos = stars.y[i]
            flux = stars.star_flux[i]
            star_stamp = stars.stamps[i]
            psf_model = self.render_psf(x=xpos,y=ypos,flux=flux)
            if type(psf_model) is galsim.image.Image:
                psf_stamp = psf_model.array
            else:
                psf_stamp = psf_model
            try:
                self.models.append(psf_model)
                self.stamps.append(psf_stamp)
                self.resids.append(star_stamp-psf_stamp)
            except:
                pdb.set_trace()
        # Do HSM fitting
        do_hsm_fit(maker=self, verbose=vb)

        # Make output quiverplot
        quiv_name = os.path.join(outdir, '_'.join([self.psf_type,'quiverplot.png']))
        quiverplot = QuiverPlot(starmaker=stars, psfmaker=self)
        quiverplot.run(scale=1, outname=quiv_name)

        # Make output star-psf residuals plot
        resid_name = os.path.join(outdir,'_'.join([self.psf_type,'flux_resid.png']))
        chi2_name = os.path.join(outdir,'_'.join([self.psf_type,'chi2.png']))
        resid_plot = ResidPlots(starmaker=stars, psfmaker=self)
        resid_plot.run(resid_name=resid_name, chi2_name=chi2_name)

        # Compute & make output rho-statistics figures
        rho_params = self.rho_params
        if rho_params == None:
            rho_params={'min_sep':200,'max_sep':5000,'nbins':10}

        self.run_rho_stats(stars=stars,rho_params=rho_params,vb=vb,outdir=outdir)

        print("finished running PSFMaker()")
        return
