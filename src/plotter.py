import psfex
import galsim,galsim.des
import treecorr
import piff
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc,rcParams
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors
import matplotlib.pyplot as plt
import os, re
import pdb, ipdb
from astropy.table import Table
import fitsio

from .utils import set_rc_params

def size_mag_plots(im_cat, star_cat, plot_name, filter_name):
    '''
    Save to file a size-magnitude plot with the stellar locus highlighted
    Inputs:
        im_cat: exposure catalog, either file name or Table() object
        star_cat : star catalog passed to PIFF, either file name or Table() object
        plot_name : plot file name
        filter_name : filter name (no support for 30mas/60mas rn)
    '''

    set_rc_params(fontsize=14)

    if type(im_cat) == str:
        image_catalog = Table.read(im_cat)
    else:
        image_catalog = im_cat

    if type(im_cat) == str:
        star_catalog = Table.read(star_cat)
    else:
        star_catalog = star_cat

    fig, axs = plt.subplots(1,2, tight_layout=True, figsize=(11,7))

    # First, do FWHM
    axs[0].plot(image_catalog['MAG_AUTO'], image_catalog['FWHM_WORLD']*3600, '.', \
            label='all objects', markersize=3)
    axs[0].plot(star_catalog['MAG_AUTO'], star_catalog['FWHM_WORLD']*3600, '.', \
            label='selected stars', markersize=3)

    #axs[0].set_xlabel(r'\texttt{MAG_AUTO}', fontsize=16)
    axs[0].set_ylabel(r'\texttt{FWHM_WORLD} (arcsec)', fontsize=16)
    axs[0].set_ylim(-0.08, 0.9)
    axs[0].set_xlim(18, 30)
    axs[0].grid(True)
    axs[0].legend(markerscale=3, fontsize=14, loc='upper left')


    # Then, flux_radius
    axs[1].plot(image_catalog['MAG_AUTO'], image_catalog['FLUX_RADIUS']*2, '.', \
            label='all objects', markersize=3)
    axs[1].plot(star_catalog['MAG_AUTO'], star_catalog['FLUX_RADIUS']*2, '.', \
            label='selected stars', markersize=3)

    axs[1].set_xlabel(r'\texttt{MAG_AUTO}', fontsize=14)
    axs[1].set_ylabel(r'2*\texttt{FLUX_RADIUS} (pix)', fontsize=14)
    axs[1].set_ylim(-0.05, 10)
    axs[1].set_xlim(18, 30)
    axs[1].grid(True)
    axs[1].legend(markerscale=3, fontsize=14, loc='upper left')

    fig.savefig(plot_name)

    return


def make_resid_plot(psf, stars, outname='star_psf_resid.png', vb=False):
    '''
    make figures of average stars, psf renderings,
    and residuals between the two

    :avg_psf : should be an instance of PSFMaker()
    :avg_star: should be an instance of StarMaker()
    '''

    set_rc_params()
    fontsize = 16
    vmin = 0.0001
    vmax = 75

    avg_stars = np.nanmean(stars.stamps,axis=0)
    avg_psfim = np.nanmean(psf.stamps,axis=0)
    avg_resid = np.nanmean(psf.resids,axis=0)

    if vb==True:
        print("avg_star total flux = %.3f" % np.sum(avg_stars))
        print("avg_psf total flux = %.3f" % np.sum(avg_psfim))

    # Calculate average sizes to display in image
    wg = (psf.hsm_sig > -9999.) & (stars.hsm_sig > -9999.)
    psf_fwhm  = np.nanmean(psf.fwhm[wg])
    psf_sigma = np.nanmean(psf.hsm_sig[wg])
    star_fwhm = np.nanmean(stars.fwhm[wg])
    star_sigma = np.nanmean(stars.hsm_sig[wg])

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15,7), tight_layout=True)

    f1 = axs[0].imshow(avg_stars, cmap=plt.cm.bwr_r,
                        norm=colors.LogNorm(vmin=vmin, vmax=vmax))
    axs[0].set_title('avg star HSM sigma = %.4f\ngs.calculateFWHM() = %.4f'
                        % (star_sigma,star_fwhm), fontsize=16)
    axs[0].axvline((avg_stars.shape[0]-1)*0.5,color='black')
    axs[0].axhline((avg_stars.shape[1]-1)*0.5,color='black')
    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(f1, cax=cax)

    f2 = axs[1].imshow(avg_psfim, cmap=plt.cm.bwr_r,
                        norm=colors.LogNorm(vmin=vmin, vmax=vmax))
    axs[1].set_title('avg PSF HSM sigma = %.4f\ngs.calculateFWHM() = %.4f'
                        % (psf_sigma,psf_fwhm), fontsize=16)
    axs[1].axvline((avg_stars.shape[0]-1)*0.5,color='black')
    axs[1].axhline((avg_stars.shape[1]-1)*0.5,color='black')

    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes("right",size="5%", pad=0.05)
    plt.colorbar(f2, cax=cax)

    resid_norm = colors.TwoSlopeNorm(0, vmin=0.9*np.min(avg_resid),
                                        vmax=0.9*np.max(avg_resid)
                                        )
    f3 = axs[2].imshow(avg_resid,norm=resid_norm, cmap=plt.cm.seismic_r)
    axs[2].set_title('sum(mean resid)= %.3f\nmean=%.2e std=%.2e' %
                        (np.nansum(avg_resid), np.nanmean(avg_resid),
                            np.nanstd(avg_resid)), fontsize=16)
    axs[2].axvline((avg_stars.shape[0]-1)*0.5,color='black')
    axs[2].axhline((avg_stars.shape[1]-1)*0.5,color='black')
    divider = make_axes_locatable(axs[2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(f3, cax=cax)

    plt.savefig(outname)

    # Write fits files out to file too
    outdir = os.path.dirname(outname)
    fout = fitsio.FITS(
            os.path.join(outdir, 'mean_star_model_resid_ims.fits'), 'rw')
    fout.write([avg_stars, avg_psfim, avg_resid],
                names=['STARS', 'MODELS', 'RESIDS']
                )

    return 0

def make_chi2_resids(psf, stars, outname='star_psf_chi2.png', vb=False):
    '''
    Make a chi2 residual plot, kinda like the flux residuals plots also made.
    '''

    return


def make_quiverplot():
    '''
    At some point, plotter should be its own class that makes instance of
    the QuiverPlot class for quiver plotting
    '''

    pass


def plot_rho_stats(rho1, rho2, rho3, rho4, rho5, pixel_scale, outname=None):
    ##
    ## rho1 correlation: dg x dg
    ##

    fontsize = 16
    set_rc_params(fontsize)


    fig,axes=plt.subplots(nrows=2,ncols=1,figsize=[10,7], sharex=True, tight_layout=True)

    r = np.exp(rho1.meanlogr) * pixel_scale / 60
    xip = np.abs(rho1.xip)
    sig = np.sqrt(rho1.varxip)

    lab1 = r'$\rho_1(\theta)$'
    lp1 = axes[0].plot(r, xip, color='tab:blue',marker='o',ls='-',label=lab1)
    axes[0].plot(r, -xip, color='tab:blue', marker='o',ls=':')
    axes[0].errorbar(r[xip>0], xip[xip>0], yerr=sig[xip>0], color='tab:blue', ls='', capsize=5)
    axes[0].errorbar(r[xip<0], -xip[xip<0], yerr=sig[xip<0], color='tab:blue', ls='', capsize=5)
    axes[0].errorbar(-r, xip, yerr=sig, color='tab:blue', capsize=5)

    #axes[0].set_xlabel(r'$\theta$ (arcmin)', fontsize=fontsize)
    axes[0].set_ylabel(r'$\xi_+(\theta)$', fontsize=fontsize)
    #axes[0].set_xscale('log')
    axes[0].set_yscale('log', nonpositive='clip')

    ##
    ## rho3 correlation: dg x dg
    ##
    r = np.exp(rho3.meanlogr) * pixel_scale / 60
    xip = np.abs(rho3.xip)
    sig = np.sqrt(rho3.varxip)

    lab3 = r'$\rho_3(\theta)$'
    lp3 = axes[0].plot(r, xip, color='tab:orange',marker='o',ls='-',label=lab3)
    axes[0].plot(r, -xip, color='tab:orange', marker='o',ls=':')
    axes[0].errorbar(r[xip>0], xip[xip>0], yerr=sig[xip>0], color='tab:orange', ls='', capsize=5)
    axes[0].errorbar(r[xip<0], -xip[xip<0], yerr=sig[xip<0], color='tab:orange', ls='', capsize=5)
    axes[0].errorbar(-r, xip, yerr=sig, color='tab:orange', capsize=5)

    ##
    ## rho4 correlation: dg x dg
    ##
    r = np.exp(rho4.meanlogr) * pixel_scale / 60
    xip = np.abs(rho4.xip)
    sig = np.sqrt(rho4.varxip)

    lab4 = r'$\rho_4(\theta)$'
    lp4 = axes[0].plot(r, xip, color='tab:green',marker='o',ls='-',label=lab4)
    axes[0].plot(r, -xip, color='tab:green', marker='o',ls=':')
    axes[0].errorbar(r[xip>0], xip[xip>0], yerr=sig[xip>0], color='tab:green', ls='', capsize=5)
    axes[0].errorbar(r[xip<0], -xip[xip<0], yerr=sig[xip<0], color='tab:green', ls='', capsize=5)
    axes[0].errorbar(-r, xip, yerr=sig, color='tab:green', capsize=5)

    axes[0].legend([lp1, lp3, lp4], fontsize=14)
    axes[0].legend(fontsize=14, loc='upper right')

    ##
    ## rho 2 correlation: g x dg
    ##
    r = np.exp(rho2.meanlogr) * pixel_scale / 60
    xip = np.abs(rho2.xip)
    sig = np.sqrt(rho2.varxip)

    lp2 = axes[1].plot(r, xip, color='tab:cyan',marker='o', ls='-', label=r'$\rho_2(\theta)$')
    axes[1].plot(r, -xip, color='tab:cyan', marker='o', ls=':')
    axes[1].errorbar(r[xip>0], xip[xip>0], yerr=sig[xip>0], color='tab:cyan', ls='', capsize=5)
    axes[1].errorbar(r[xip<0], -xip[xip<0], yerr=sig[xip<0], color='tab:cyan', ls='', capsize=5)
    axes[1].errorbar(-r, xip, yerr=sig, color='tab:cyan', capsize=5)

    axes[1].set_xlabel(r'$\theta$ (arcmin)', fontsize=fontsize)
    axes[1].set_ylabel(r'$\xi_+(\theta)$', fontsize=fontsize)
    axes[1].set_xscale('log')
    axes[1].set_yscale('log', nonpositive='clip')

    ##
    ## rho5 correlation
    ##
    r = np.exp(rho5.meanlogr) * pixel_scale / 60.
    xip = rho5.xip
    sig = np.sqrt(rho5.varxip)

    lp5 = axes[1].plot(r, xip, color='tab:purple',marker='o', ls='-', label=r'$\rho_5(\theta)$')
    axes[1].plot(r, -xip, color='tab:purple', marker='o', ls=':',)
    axes[1].errorbar(r[xip>0], xip[xip>0], yerr=sig[xip>0], color='tab:purple', ls='', capsize=5)
    axes[1].errorbar(r[xip<0], -xip[xip<0], yerr=sig[xip<0], color='tab:purple', ls='', capsize=5)
    axes[1].errorbar(-r, xip, yerr=sig, color='tab:purple', capsize=5)

    axes[1].legend([lp2,lp5])
    plt.legend(loc='upper right')

    fig.savefig(outname)

    return

def compare_rho_stats(prefix, pixel_scale, file_path='./', rho_files=None):
    '''
    Make rho ratio plots for different PSF types.
    Note that the master_psf_diagnostics.py file nomenclature is assumed:
    [psf_type]_rho_[1-5].txt with psf_type={'epsfex', 'gpsfex', 'piff'}
    '''

    set_rc_params(fontsize=15)
    plt.style.use('dark_background')
    plt.rcParams.update({'figure.facecolor':'w'})

    print(f'Looking for rho-stat files in {file_path}')

    for i in range(1,6):

        try:
            pexn=os.path.join(file_path,''.join(['epsfex_rho_',str(i),'.txt']))
            pex=Table.read(pexn,format='ascii',header_start=1)
        except:
            print('no pex found')
            pex=None
        try:
            gpsfn = os.path.join(file_path,''.join(['gpsfex_rho_',str(i),'.txt']))
            gpsf=Table.read(gpsfn,format='ascii',header_start=1)
        except:
            print('no gpsf found')
            gpsf=None
        try:
            piffn = os.path.join(file_path,''.join(['piff_rho_',str(i),'.txt']))
            piff=Table.read(piffn,format='ascii',header_start=1)
        except:
            print('no piff found')
            piff=None

        savename = os.path.join(file_path,'rho_{}_comparisons.png'.format(i))

        fig,ax=plt.subplots(nrows=1,ncols=1,figsize=[10,6])
        ax.set_xscale('log')
        ax.set_yscale('log', nonpositive='clip')
        ax.set_ylabel(r'$\rho_{}(\theta)$'.format(i))
        ax.set_xlabel(r'$\theta$ (arcsec)')
        legends = []

        if pex is not None:

            r = pex['meanr'] * pixel_scale / 60 # from pixels --> arcminutes
            pex_xip = np.abs(pex['xip'])
            pex_sig = pex['sigma_xip']

            lp = ax.plot(r, pex_xip, color='C3', marker='o', ls='-', lw=2,
                            label='pex')
            ax.plot(r, -pex_xip, color='C3',  marker='o',ls=':')
            ax.errorbar(r[pex_xip>0], pex_xip[pex_xip>0], color='C3',
                            yerr=pex_sig[pex_xip>0], capsize=5, ls='')
            ax.errorbar(r[pex_xip<0], -pex_xip[pex_xip<0], color='C3',
                            yerr=pex_sig[pex_xip<0], capsize=5, ls='')
            ax.errorbar(-r, pex_xip, yerr=pex_sig, capsize=5, color='C3')

            legends.append(lp)

        if gpsf is not None:

            r = gpsf['meanr'] * pixel_scale / 60 # from pixels --> arcminutes
            gpsf_xip = np.abs(gpsf['xip'])
            gpsf_sig = gpsf['sigma_xip']

            lp2 = ax.plot(r, gpsf_xip, color='C4', marker='o', lw=2,
                            ls='-', label='gpsf')
            ax.plot(r, -gpsf_xip, color='C4', lw=2, marker='o',ls=':')
            ax.errorbar(r[gpsf_xip>0], gpsf_xip[gpsf_xip>0], yerr=gpsf_sig[gpsf_xip>0],
                            capsize=5, color='C4', ls='')
            ax.errorbar(r[gpsf_xip<0], -gpsf_xip[gpsf_xip<0], yerr=gpsf_sig[gpsf_xip<0],
                            capsize=5, color='C4', ls='')
            ax.errorbar(-r, gpsf_xip, yerr=gpsf_sig, capsize=5, color='C4')

            legends.append(lp2)

        if piff is not None:

            r = piff['meanr'] * pixel_scale / 60 # from pixels --> arcminutes
            piff_xip = np.abs(piff['xip'])
            piff_sig = piff['sigma_xip']

            lp3 = ax.plot(r, piff_xip, color='C5', marker='o',lw=2,
                            ls='-', label='piff')
            ax.plot(r, -piff_xip, color='C5',  marker='o', lw=2, ls=':')
            ax.errorbar(r[piff_xip>0], piff_xip[piff_xip>0], color='C5',
                            yerr=piff_sig[piff_xip>0], capsize=5, ls='')
            ax.errorbar(r[piff_xip<0], -piff_xip[piff_xip<0], color='C5',
                            yerr=piff_sig[piff_xip<0], capsize=5,  ls='')
            ax.errorbar(-r, piff_xip, yerr=piff_sig, capsize=5, color='C5')

            legends.append(lp3)

        plt.legend()
        fig.savefig(savename)

    print("rho comparison plots done")

    return
