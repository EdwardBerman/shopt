###
### This is a quick and dirty script to get PSF models
### for JWST data in "i2d" format.
###
import numpy as np
import os, re
from astropy.io import fits
import pdb
from astropy.table import Table, vstack, hstack
import glob
from esutil import htm
from argparse import ArgumentParser
import ipdb, pdb
from src.plotter import size_mag_plots
from src.utils import read_yaml
from src.box_cutter import BoxCutter
import fitsio


def parse_args():

    parser = ArgumentParser()

    parser.add_argument('images', nargs='+',
                        help = 'Images to process (wildcards OK)')
    parser.add_argument('-outdir', default=None,
                        help = 'Where to save files')
    parser.add_argument('-config', default=None,
                        help = 'configuration file')
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help = 'Overwrite sci/weight files')
    parser.add_argument('--vb', action='store_true', default=True,
                        help = 'Print detailed messages [does nothing for now]')

    return parser.parse_args()


def get_star_params(config=None):
    '''
    Return a look-up table containing approximate stellar
    FWHMs for each of the filters based on DEC2022 COSMOS-Web simulations
    This is horrendous coding and should be fixed ASAP -- try a yaml?

    Here, selecting on FLUX_RADIUS parameter in pixels. The longer wavelength
    images have a flux_rad vs. flux_auto
    '''

    filter_names = ['f115w','f150w','f277w', 'f444w']
    fwhms = [0.058, 0.0628, 0.125, 0.165]
    min_size = [0.88, 1, 1.83, 2.33]
    max_size = [1.7, 1.5, 2.5, 3.2]
    max_mag = [26.2, 26.5, 27, 27.1]
    min_mag = [19.5, 19.5, 19, 19]

    star_params = Table([filter_names, fwhms, min_size,\
                        max_size, max_mag, min_mag],
                        names=['filter_name', 'star_fwhm', 'min_size',\
                        'max_size', 'max_mag', 'min_mag']
                        )

    return star_params


def run_sextractor(image_file, weight_file, config, star_params):
    '''
    Run Source Extractor, including the appropriate star FWHM for
    preliminary star identification.

    Inputs
        image_file: image for SExtractor
        weight_file: weight file for SExtractor
        configdir: directory of SEx configuration files
        outdir: directory for outputs
        star_params: table with appx. size of stars in image
    '''

    sci = config['sci_image']['hdu']
    wht = config['weight_image']['hdu']
    outdir = config['outdir']
    configdir = config['configdir']

    img_basename = os.path.basename(image_file)

    cat_name = os.path.join(
                outdir, img_basename.replace('i2d.fits','cat.fits')
                )
    bkg_sub  = os.path.join(
                    outdir, img_basename.replace('i2d.fits','sub.fits')
                    )
    sgm_name = os.path.join(
                outdir, img_basename.replace('i2d.fits','sgm.fits')
                )
    filter_name = re.search(r"f(\d){3}w", img_basename).group()

    wg = np.isin(star_params['filter_name'], filter_name)
    star_fwhm = star_params[wg]['star_fwhm']
    star_fwhm = np.float64(star_fwhm)

    image_arg  = f'"{image_file}[{sci}]"'
    seeing_arg = f'-SEEING_FWHM {star_fwhm}'
    weight_arg = f'-WEIGHT_IMAGE "{weight_file}[{wht}]"'
    name_arg   = f'-CATALOG_NAME {cat_name}'
    check_arg  = f'-CHECKIMAGE_NAME  {bkg_sub},{sgm_name}'
    param_arg  = '-PARAMETERS_NAME ' + os.path.join(configdir, 'sextractor.param')
    nnw_arg    = '-STARNNW_NAME ' + os.path.join(configdir,'default.nnw')
    filter_arg = '-FILTER_NAME ' +  os.path.join(configdir,'gauss_3.0_5x5.conv')
    config_arg = '-c ' + os.path.join(configdir, 'sextractor.config')

    cmd = ' '.join([
            'sex', image_arg, weight_arg, name_arg, check_arg, param_arg,
                nnw_arg, filter_arg, seeing_arg, config_arg
                ])
    print("sex cmd is " + cmd)
    os.system(cmd)

    return


def make_starcat(image_file, config, star_params=None, thresh=0.85):
    '''
    Create a star catalog from the SExtractor image catalog using cuts on
    the SExtractor CLASS_STAR parameter and the supplied table of star properties.
    Alternatively, can match against a reference star catalog (truthstars).
    Also plot size-magnitude diagram for image and star catalogs.

    Inputs:
        image_file : image file
        out_dir : where to save star catalog (also assumed to be location of image catalog)
        truthstars : reference star catalog
        star_params : table with stellar locus parameters
        thresh : CLASS_STAR threshold for accepting as star

    Outputs:
        star_cat_file: name of the star catalog (saved to file)

    TO DO:
        - make starparams a config file. Alternatively, improve auto-star selection
    '''

    img_basename = os.path.basename(image_file)
    outdir = config['outdir']

    try:
        filter_name = re.search(r"f(\d){3}w", image_file).group() \
                        +'_'+ re.search(r"(\d){2}mas",image_file).group()
    except AttributeError:
        filter_name = re.search(r"f(\d){3}w", image_file).group()

    imcat_name = os.path.join(
                outdir, img_basename.replace('i2d.fits','cat.fits')
                )
    starcat_name = os.path.join(
                outdir, img_basename.replace('i2d.fits','starcat.fits')
                )
    plot_name = os.path.join(
                outdir, img_basename.replace('i2d.fits','sizemag.pdf')
                )

    if not os.path.exists(imcat_name):
        print(f'\n\ncould not find image im_cat file {imcat_name}\n\n')
    else:
        im_cat_fits = fits.open(imcat_name)
        im_cat = im_cat_fits['LDAC_OBJECTS'].data

    if config['star_params']['truthstars'] is not None:
        truth_star_tab = Table.read(config['star_params']['truthstars'])
        truthmatch = htm.Matcher(16, ra=truth_star_tab['x_or_RA'],
                                    dec=truth_star_tab['y_or_Dec'])
        cat_ind, truth_ind, dist = truthmatch.match(
                                    ra=im_cat['ALPHAWIN_J2000'],
                                    dec=im_cat['DELTAWIN_J2000'],
                                    maxmatch=1, radius = 0.5/3600)

    else:

        filter_name = re.search(r"f(\d){3}w", image_file).group()
        wg = np.isin(star_params['filter_name'], filter_name)
        min_size = star_params[wg]['min_size']
        min_size = np.float64(min_size)
        max_size = star_params[wg]['max_size']
        max_size = np.float64(max_size)
        max_mag = star_params[wg]['max_mag']
        max_mag = np.float64(max_mag)
        min_mag = star_params[wg]['min_mag']
        min_mag = np.float64(min_mag)

        star_selec = (im_cat['CLASS_STAR'] > thresh) \
                        & (im_cat['MAG_AUTO'] < max_mag) \
                        & (im_cat['MAG_AUTO'] > min_mag) \
                        & (im_cat['FLUX_RADIUS'] > min_size) \
                        & (im_cat['FLUX_RADIUS'] < max_size)

        cat_ind = np.arange(len(star_selec))[star_selec]

    star_cat = im_cat[cat_ind]

    im_cat_fits['LDAC_OBJECTS'].data = star_cat
    im_cat_fits.writeto(starcat_name, overwrite=True)

    # Make size-mag plot
    size_mag_plots(im_cat, star_cat, plot_name, filter_name)

    return starcat_name


def add_err_cutout(boxcut, image_file, starcat_file):
    '''
    Wrapper to call BoxCutter.grab_boxes and add an extra ERR stamp for
    chi2 calculations. Adding the extra column to the FITS HDU List is a bit
    roundabout, but I couldn't figure out a better way to do it.

    Inputs
        grabber: should be a BoxCutter instance
        err_file: the error file to add to catalog
        starcat_name: catalog that's going to get an extra vignet/stamp
        outdir: where is it getting saved?
    '''

    # V. nice, accessing config through imported class
    config = boxcut.config
    cat_hdu = config['input_catalog']['hdu']
    box_size = boxcut.box_size

    # Read in the fits file so that we can add the column ourselves
    sc_fits = fitsio.FITS(starcat_file, 'rw')
    starcat = sc_fits[cat_hdu].read()

    # Call to grab_boxes method
    errs = boxcut.grab_boxes(image_file=image_file, cat_file=starcat)
    data = np.zeros(len(errs), dtype=[('ERR_VIGNET', \
                        'f4', (box_size, box_size))])
    for i, err in enumerate(errs):
        data['ERR_VIGNET'][i] = err
    sc_fits[cat_hdu].insert_column('ERR_VIGNET', data['ERR_VIGNET'])
    sc_fits.close()

    return


def run_piffy(image_file, starcat_file, config, echo=True):
    '''
    Run PIFF using supplied im_file and star catalog!

    Inputs:
        im_file : the exposure to characterize
        star_cat_file : catalog of stars for PSF fitting
        configdir : path to PIFF config file
        outdir : where to save PIFF results
    '''

    sci = config['sci_image']['hdu']

    # Get the image filename root to name outputs
    base_name   = os.path.basename(image_file)

    output_name = base_name.replace('.fits','.piff')

    piff_outdir = os.path.join(config['outdir'], \
                    'piff-output', base_name.split('.')[0])

    bkg_sub = os.path.join(config['outdir'],
                base_name.replace('i2d.fits', 'sub.fits'))

    # PIFF wants RA in hours, not degrees
    ra = fits.getval(image_file, 'CRVAL1', ext=sci)/15.0
    dec = fits.getval(image_file, 'CRVAL2', ext=sci)

    # Load PIFF configuration file
    piff_config_arg = os.path.join(config['configdir'], 'piff.config')

    # Now run PIFF on that image and accompanying catalog
    image_arg   = f'input.image_file_name={image_file}'
    coord_arg   = f'input.ra={ra} input.dec={dec}'
    psfcat_arg  = f'input.cat_file_name={starcat_file}'
    output_arg  = f'output.file_name={output_name} output.dir={piff_outdir}'

    cmd = ' '.join([
             'piffify', piff_config_arg, image_arg, coord_arg, \
                psfcat_arg, output_arg
                ])
    print('piff cmd is ' + cmd)

    if echo is False:
        os.system(cmd)

    return


def run_psfex(image_file, starcat_file, config):
    '''
    Run PSFEx, creating an output directory if one doesn't already exist.
    Default is to create one directory per exposure.
    '''

    base_name = os.path.basename(image_file)
    outcat_name  = starcat_file.replace('cat.fits', 'psfex_cat.fits')
    psfex_outdir = os.path.join(config['outdir'], \
                    'psfex-output', base_name.split('.')[0])
    if not os.path.isdir(psfex_outdir):
        os.system(f'mkdir -p {psfex_outdir}')

    psfex_config_arg = '-c ' + os.path.join(config['configdir'],'psfex.config')
    outcat_arg = f'-OUTCAT_NAME {outcat_name}'
    outdir_arg = f'-PSF_DIR {psfex_outdir}'

    cmd = ' '.join(['psfex',
                starcat_file, psfex_config_arg, outcat_arg, outdir_arg]
            )

    print(f'psfex cmd is {cmd}')
    os.system(cmd)

    # Again, if there's a better way to do this, I don't know it.
    cleanup_cmd = ' '.join([f'mv *_{os.path.basename(starcat_file)}',
        f'*_{os.path.basename(starcat_file).replace(".fits", ".pdf")} *.xml',
        psfex_outdir])
    os.system(cleanup_cmd)

    # And for convenience... save a PSFEx stars-only file
    pexcat = Table.read(outcat_name, hdu=2)
    starcat = Table.read(starcat_file, hdu=2)
    pexstar_name = os.path.join(config['outdir'],
                    base_name.replace('i2d.fits', 'pex_stars.fits'))

    pex_stars = pexcat['FLAGS_PSF']==0
    starcat[pex_stars].write(pexstar_name, format='fits', overwrite=True)

    return


def main(args):

    i2d_images = args.images
    overwrite = args.overwrite
    vb = args.vb

    # This needs to be its own function, I think
    config = read_yaml(args.config)
    configdir = config['configdir']
    outdir = config['outdir']

    # Set default output directory values if none provided
    if configdir is None:
        config['configdir'] = \
                '/Users/j.mccleary/Research/jwst_cosmos/cweb_psf/astro_config/'

    if outdir is None:
        basedir = os.path.commonpath(images)
        outdir = os.path.join(basedir,'working')
        config['outdir'] = os.path.join(basedir,'working')

    if not os.path.isdir(outdir):
        cmd = 'mkdir -p {outdir}'.format(outdir=outdir)
        os.system(cmd)
        print(f'Made output directory {outdir}')
    else:
        print(f'Output directory {outdir} exists, continuing...')

    # Placeholder: table of stellar locus parameters
    star_params = get_star_params()

    # Create a BoxCutter instance
    boxcut = BoxCutter(config_file=args.config)

    # Process exposures
    for i2d in i2d_images:

        print(f'Working on file {i2d}...\n\n')

        image_file = i2d

        run_sextractor(image_file=image_file, weight_file=image_file,
                            config=config, star_params=star_params)

        starcat_file = make_starcat(image_file=image_file, config=config,
                                        star_params=star_params)

        add_err_cutout(boxcut, image_file=image_file,
                                starcat_file=starcat_file)

        run_psfex(image_file, starcat_file=starcat_file,
                    config=config)

        run_piffy(image_file, starcat_file=starcat_file,
                    config=config, echo=True)

    return 0


if __name__ == '__main__':

    args = parse_args()
    rc = main(args)

    if rc == 0:
        print('get_jwst_psfs.py has completed succesfully')
    else:
        print(f'get_jwst_psfs.py has failed w/ rc={rc}')
