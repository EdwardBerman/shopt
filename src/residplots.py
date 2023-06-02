import unittest
test = unittest.TestCase
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc,rcParams
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors
import matplotlib.pyplot as plt
import ipdb, pdb
from astropy.io import fits
from scipy.stats import chi2
import galsim

from .utils import AttrDict, set_rc_params

class ResidPlots:
    '''
    Resid plots were getting messy, so made it a class!
    Ideally should define a plotter class that gets inherited by
    '''

    def __init__(self, starmaker, psfmaker):
        '''
        Attributes
            stars:  instance of StarMaker
            psf:  instance of PSFMaker
            scale:  scaling for quiverplots

        test.assertIsInstance(self, starmaker, src.starmaker.StarMaker,
                        msg='starmaker must be instance of StarMaker class'
                        )
        test.assertIsInstance(self, psfmaker, src.psfmaker.PSFMaker,
                         msg='psfmaker must be instance of PSFMaker class'
                         )
        '''
        self.stars = starmaker
        self.psfs  = psfmaker

        self.star_dict = {}
        self.psf_dict = {}
        self.resid_dict = {}
        self.chi2_dict = {}


    def _make_im_dict(self, maker, stamps, wg):
        '''
        Calculate ellipticity parameters using input HSM moments

        Inputs
                maker:  should be an instance of either StarMaker or PSFMaker
                stamps: either list of arrays or list of Galsim Image instances
                wg:  where star and psf fits both succeeded
        Returns:
                ellip_dict: dict with e1, e2, and theta for plotting,
                            plus some summary stats, cast to a class
        '''

        fwhm = np.nanmedian(maker.fwhm[wg])
        sigma = np.nanmedian(maker.hsm_sig[wg])

        try:

            if type(stamps[0]) is galsim.image.Image:
                stamp_arr = []
                for stamp in stamps:
                    stamp_arr.append(stamp.array)
                avg_im = np.nanmedian(stamp_arr, axis=0)
            else:
                avg_im = np.nanmedian(stamps, axis=0)

        except:
            pdb.set_trace()

        im_dict = dict(avg_im = avg_im,
                        fwhm  = fwhm,
                        sigma = sigma
                        )

        return AttrDict(im_dict)


    def _populate_dicts(self):
        '''
        Populate the ellipticity dictionaries for plotting
        '''

        psfs  = self.psfs
        stars = self.stars

        wg = (psfs.hsm_g1 > -9999) & (stars.hsm_g1 > -9999)
        self.star_dict  = self._make_im_dict(stars, stars.stamps, wg)
        self.psf_dict   = self._make_im_dict(psfs, psfs.stamps, wg)
        self.resid_dict = self._make_im_dict(psfs, psfs.resids, wg)

        return


    def make_chi2(self, nparams=2, outname='chi2_residuals.png'):
        '''
        Compute the chi-squared for each image. Loop over residuals to get
        chi-squared for each stamp, then create a mean (or maybe median)
        '''
        psf = self.psfs
        star = self.stars

        npix = psf.vignet_size * psf.vignet_size
        dof = npix * psf.sample_scale * psf.sample_scale

        chi2_maps = []

        for i, resid in enumerate(psf.resids):
            noise_map = star.err_stamps[i]
            #normed_noise_map = noise_map / np.nansum(noise_map)
            chi2_map = np.square(np.divide(resid, noise_map))
            reduced_chi2_map = chi2_map / dof
            chi2_maps.append(reduced_chi2_map)

        # masked_chi2 = np.ma.masked_where(np.isinf(chi2_maps), chi2_maps)
        masked_chi2 = np.ma.masked_invalid(chi2_maps)

        # Average (median) image
        avg_chi2_im = np.ma.median(masked_chi2, axis=0).data

        # Total chi2
        chi_square = np.ma.sum(masked_chi2)

        # Calculate reduced chi2
        ddof = len(chi2_maps)
        reduced_chi_square = chi_square / ddof

        # Calculate p-value
        p_value = 1 - chi2.cdf(chi_square, dof)

        # get a dict with all those values!
        chi2_dict = dict(avg_im = avg_chi2_im,
                            reduced_chi_square = reduced_chi_square,
                            p_value = p_value
                            )
        self.chi2_dict = AttrDict(chi2_dict)

        # Save the chi image to a fits file, too
        im = fits.PrimaryHDU(np.sqrt(avg_chi2_im))

        for key in list(chi2_dict.keys())[1:]:
            im.header.set(key, chi2_dict[key])
        im.writeto(outname.replace('.png', '.fits'), overwrite=True)

        return


    def _make_mpl_dict(self, index, vmin=None, vmax=None, avg_im=None):
        '''
        EXTREMELY SPECIFIC PLOTTING KEYWORDS -- CHANGE WITH CAUTION
        (Plots may not make sense or errors may be thrown). Assumes that passed
        avg_im is a residual plot of some sort.

            #norm = colors.TwoSlopeNorm(np.median(avg_im),
            #        vmin=0.8*np.min(avg_im),
            #        vmax=0.8*np.max(avg_im))

        I used to use colors.TwoSlopeNorm and the seismic_r color map for the
        flux residuals, but have decided to go with SymLogNorm for now.
        '''
        if (avg_im is not None):

            norm = colors.SymLogNorm(linthresh=1e-4,
                            vmin=np.min(avg_im),
                            vmax=np.max(avg_im))
            cmap = plt.cm.bwr_r

        else:
            if (np.min(self.star_dict.avg_im) <=0):
                vmin = 0.001
            else:
                vmin = np.min(self.star_dict.avg_im)

            norm = colors.SymLogNorm(vmin=vmin,
                            vmax=np.max(self.star_dict.avg_im),
                            linthresh=1e-4)
            cmap=plt.cm.turbo

        mpl_dict = dict(cmap=cmap, norm=norm)

        return mpl_dict


    def _get_plot_titles(self):

        sd = self.star_dict
        pd = self.psf_dict
        rd = self.resid_dict
        xd = self.chi2_dict

        star_title = 'median HSM $\sigma^{*} = %.4f$ pix\ngs.calculateFWHM() = %.4f$^{\prime\prime}$'\
                    % (sd.sigma, sd.fwhm)
        psf_title = 'median HSM $\sigma^{PSF} = %.4f$ pix\ngs.calculateFWHM() = %.4f$^{\prime\prime}$'\
                    % (pd.sigma, pd.fwhm)
        resid_title = 'sum(median resid)= %.3f\nmedian=%1.3e std=%1.3e'\
                    % (np.nansum(rd.avg_im), np.nanmedian(rd.avg_im),
                        np.nanstd(rd.avg_im))
        chi2_title = 'Total $\chi^2_{dof} = %.2f$\n'\
                    % (xd.reduced_chi_square)

        sd.title = star_title; pd.title = psf_title
        rd.title = resid_title; xd.title = chi2_title

        return


    def _make_fig(self, dicts, mpl_dicts):
        '''
        Generic method to make residuals plots
        '''

        # First things first I'm the reallest
        set_rc_params(fontsize=16)

        fig, axs = plt.subplots(nrows=1, ncols=3, sharey=True,
                                    figsize=[15,7], tight_layout=True)
        for i, dc in enumerate(dicts):
            im = axs[i].imshow(dc.avg_im, **mpl_dicts[i])
            axs[i].set_title(dc.title)
            divider = make_axes_locatable(axs[i])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax)
            axs[i].axvline((dc.avg_im.shape[0]-1)*0.5,color='black')
            axs[i].axhline((dc.avg_im.shape[1]-1)*0.5,color='black')

        return fig


    def make_flux_resid_plot(self, outname='flux_residuals.png'):
        '''
        Make Flux residual image
        '''

        dicts = [self.star_dict, self.psf_dict, self.resid_dict]

        mpl_dicts=[]
        for i, dct in enumerate(dicts):
            if i==2:
                mpl_dict = dict(norm=colors.LogNorm(), cmap=plt.cm.bwr_r)

            else:
                star_norm = colors.SymLogNorm(linthresh=1e-4)
                mpl_dict = dict(norm=star_norm, cmap=plt.cm.turbo)

            mpl_dicts.append(mpl_dict)

        # Make actual plot
        fig = self._make_fig(dicts, mpl_dicts)

        # Save plot
        fig.savefig(outname)


    def make_chi2_plot(self, outname='chi2_residuals.png'):
        '''
        Make Chi-squared residual image
        '''

        dicts = [self.star_dict, self.psf_dict, self.chi2_dict]

        mpl_dicts=[]
        for i, dct in enumerate(dicts):
            if i==2:
                mpl_dict = dict(norm=colors.LogNorm(), cmap=plt.cm.gist_ncar)

            else:
                star_norm = colors.SymLogNorm(linthresh=1e-4)
                mpl_dict = dict(norm=star_norm, cmap=plt.cm.turbo)

            mpl_dicts.append(mpl_dict)

        # Make actual plot
        fig = self._make_fig(dicts, mpl_dicts)

        # Save it
        fig.savefig(outname)


    def run(self, resid_name=None, chi2_name=None):
        '''
        Make flux and residuals plots
        '''

        # Populate dicts
        self._populate_dicts()

        # Make chi-square residuals;
        self.make_chi2(nparams=3, outname=chi2_name)

        # Get titles (they're defined here!)
        self._get_plot_titles()

        # Make flux residual plots
        self.make_flux_resid_plot(resid_name)

        # And make the chi squared plot
        self.make_chi2_plot(chi2_name)

        return
