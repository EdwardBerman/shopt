import matplotlib.pyplot as plt
from matplotlib import rc,rcParams
from esutil import htm
import astropy.wcs as wcs
import yaml
import re

class AttrDict(dict):
    '''
    More convenient to access dict keys with dict.key than dict['key'],
    so cast the input dict into a class!
    '''

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def set_rc_params(fontsize=None):
    '''
    Set figure parameters
    '''

    if fontsize is None:
        fontsize=16
    else:
        fontsize=int(fontsize)

    rc('font',**{'family':'serif'})
    rc('text', usetex=True)

    plt.rcParams.update({'figure.facecolor':'w'})
    plt.rcParams.update({'axes.linewidth': 1.3})
    plt.rcParams.update({'xtick.labelsize': fontsize})
    plt.rcParams.update({'ytick.labelsize': fontsize})
    plt.rcParams.update({'xtick.major.size': 8})
    plt.rcParams.update({'xtick.major.width': 1.3})
    plt.rcParams.update({'xtick.minor.visible': True})
    plt.rcParams.update({'xtick.minor.width': 1.})
    plt.rcParams.update({'xtick.minor.size': 6})
    plt.rcParams.update({'xtick.direction': 'out'})
    plt.rcParams.update({'ytick.major.width': 1.3})
    plt.rcParams.update({'ytick.major.size': 8})
    plt.rcParams.update({'ytick.minor.visible': True})
    plt.rcParams.update({'ytick.minor.width': 1.})
    plt.rcParams.update({'ytick.minor.size':6})
    plt.rcParams.update({'ytick.direction':'out'})
    plt.rcParams.update({'axes.labelsize': fontsize})
    plt.rcParams.update({'axes.titlesize': fontsize})
    plt.rcParams.update({'legend.fontsize': int(fontsize-2)})

    return

def match_coords(cat1, cat2):
    '''
    Utility function to match cat1 to cat 2 using celestial coordinates
    '''

    # Either 'ra/dec' or 'ALPHAWIN_J2000/DELTAWIN_J2000'!

    if 'ra' in cat1.colnames:
        cat1_ra = cat1['ra']
        cat1_dec =  cat1['dec']
    elif 'ALPHAWIN_J2000' in cat1.colnames:
        cat1_ra = cat1['ALPHAWIN_J2000']
        cat1_dec =  cat1['DELTAWIN_J2000']
    else:
        raise KeyError('non-standard RA/Dec column in cat1')

    if 'ra' in cat2.colnames:
        cat2_ra = cat2['ra']
        cat2_dec =  cat2['dec']
    elif 'ALPHAWIN_J2000' in cat2.colnames:
        cat2_ra = cat2['ALPHAWIN_J2000']
        cat2_dec =  cat2['DELTAWIN_J2000']
    else:
        raise KeyError('non-standard RA/Dec column in cat2')

    cat1_matcher = htm.Matcher(16, ra=cat1_ra, dec=cat1_dec)

    cat2_ind, cat1_ind, dist = cat1_matcher.match(ra=cat2_ra,
                                                  dec=cat2_dec,
                                                  maxmatch=1,
                                                  radius=0.5/3600.
                                                  )
    if self.vb == True:
        print(f'{len(dist)}/{len(cat1)} gals matched to truth')

    return cat1[cat1_ind], cat2[cat2_ind]

def read_yaml(yaml_file):
    '''
    current package has a problem reading scientific notation as
    floats; see
    https://stackoverflow.com/questions/30458977/yaml-loads-5e-6-as-string-and-not-a-number
    '''

    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
        [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))

    with open(yaml_file, 'r') as stream:
        # return yaml.safe_load(stream) # see above issue
        return yaml.load(stream, Loader=loader)
