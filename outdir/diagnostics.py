from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cbook as cbook
from matplotlib import cm
from matplotlib import animation
import imageio
from mpl_toolkits.axes_grid1 import make_axes_locatable


plt.ion()

f = fits.open('/home/eddieberman/research/mcclearygroup/shopt/outdir/summary.shopt')

polyMatrix = f[0].data

def polynomial_interpolation_star(u,v, polynomialMatrix):
    r,c = np.shape(f[0].data)[0], np.shape(f[0].data)[1]
    star = np.zeros((r,c))
    for i in range(r):
        for j in range(c):
            star[i,j] = polynomialMatrix[i,j][0]*u**3 + \
                        polynomialMatrix[i,j][1]*v**3 + \
                        polynomialMatrix[i,j][2]*u**2*v + \
                        polynomialMatrix[i,j][3]*u*v**2 + \
                        polynomialMatrix[i,j][4]*u**2 + \
                        polynomialMatrix[i,j][5]*v**2 + \
                        polynomialMatrix[i,j][6]*u*v + \
                        polynomialMatrix[i,j][7]*u + \
                        polynomialMatrix[i,j][8]*v + \
                        polynomialMatrix[i,j][9]
    star = star/np.sum(star)
    return star

a = polynomial_interpolation_star(f[1].data['u coordinates'][0], f[1].data['v coordinates'][0]   ,polyMatrix)
print(np.shape(a))
fig, axes = plt.subplots(1, 3)

# Display the first image in the first subplot
axes[0].imshow(f[4].data[0, :, :  ], norm=colors.SymLogNorm(linthresh=1*10**(-4)))
axes[0].set_title('Pixel Grid Fit')

# Display the second image in the second subplot
axes[1].imshow(a, norm=colors.SymLogNorm(linthresh=1*10**(-4)))
axes[1].set_title('Polynomial Interpolation')

axes[2].imshow(f[4].data[3, :, :  ] - a, norm=colors.SymLogNorm(linthresh=1*10**(-4)))
axes[2].set_title('Residuals')

# Adjust the spacing between subplots
plt.tight_layout()
plt.show()



vignets = f[2].data
a,b,c = np.shape(f[4].data)
pixelGrid = np.zeros((b,c,a))

for i in range(a):
    pixelGrid[:,:,i] = f[4].data[i,:,:]

print(np.shape(pixelGrid))
print(np.shape(vignets))

fig2, axes2 = plt.subplots(1, 3)
axes2[0].imshow(vignets[:,:,0], norm=colors.SymLogNorm(linthresh=1*10**(-4)))
axes2[0].set_title('vignets')
axes2[1].imshow(pixelGrid[:,:,0], norm=colors.SymLogNorm(linthresh=1*10**(-4)))
axes2[1].set_title('pixel grid')
axes2[2].imshow(vignets[:,:,0] - pixelGrid[:,:,0], norm=colors.SymLogNorm(linthresh=1*10**(-4)))
axes2[2].set_title('Log Scale Residuals')
plt.tight_layout()
plt.show()


def meanRelativeError(vignets, pixelGrid):
    meanRelativeError = np.zeros((vignets.shape[0], vignets.shape[1]))
    for j in range(vignets.shape[0]):
        for k in range(vignets.shape[1]):
            RelativeError = []
            for i in range(vignets.shape[2]):
                RelativeError.append((vignets[j,k,i] - pixelGrid[j,k,i]) / (vignets[j,k,i]) + 1e-10)
            meanRelativeError[j,k] = np.mean(RelativeError)
    return meanRelativeError

fig3, axes3 = plt.subplots(1)
im = axes3.imshow(meanRelativeError(vignets, pixelGrid), cmap=plt.cm.bwr_r, norm=colors.SymLogNorm(linthresh=1*10**(-4)))
axes3.set_title('Mean Relative Error')
divider = make_axes_locatable(axes3)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig3.colorbar(im, cax=cax)

fft_image = np.fft.fft(vignets[:,:,0] - pixelGrid[:,:,0])
fft_image = np.abs(fft_image) ** 2

pk = []
for i in range(1, 11):
    radius = np.linspace(1, max(vignets.shape[0]/2, vignets.shape[1]/2) - 1, num=10)
    radiusPixels = []
    for u in range(fft_image.shape[0]):
        for v in range(fft_image.shape[1]):
            if round(np.sqrt((u - fft_image.shape[0]/2)**2 + (v - fft_image.shape[1]/2)**2) - radius[i-1]) == 0:
                radiusPixels.append(fft_image[u, v])
    pk.append(np.mean(radiusPixels))


fig4, axes4 = plt.subplots(1,2)
axes4[0].imshow(fft_image)
axes4[0].set_title('FFT of Residuals')
axes4[1].plot(radius, pk)
axes4[1].set_title('Power Spectra')

def update(frame):
    im.set_array(polyMatrix[:, :, frame]) # Update the plot data for each frame
    return im,

fig5 = plt.figure()
ax5 = fig5.add_subplot(111)
im = ax5.imshow(polyMatrix[:, :, 0], cmap='viridis', norm=colors.SymLogNorm(linthresh=1*10**(-4)), interpolation='nearest')

frames = range(np.shape(polyMatrix)[2])  # Number of frames
animation = animation.FuncAnimation(fig5, update, frames=frames, interval=200, blit=True)

def mreHist(vignets, pixelGrid):
    RelativeError = []
    for j in range(vignets.shape[0]):
        for k in range(vignets.shape[1]):
            for i in range(vignets.shape[2]):
                RelativeError.append((vignets[j,k,i] - pixelGrid[j,k,i]) / (vignets[j,k,i]) + 1e-10)
    return RelativeError

fig6, axes6 = plt.subplots(1, figsize=(30, 10))
bins = np.linspace(-1, 1, 21)
axes6.hist(mreHist(vignets, pixelGrid), bins=bins, color = "lightblue", ec="red", lw=3)
axes6.set_xlabel('Relative Error', fontsize=20)
axes6.set_ylabel('Frequency',fontsize=20)
axes6.set_yscale('log')
axes6.set_title(f'Relative Error Histogram (logscale)\n $\mu = {np.mean(mreHist(vignets, pixelGrid))}$,  $\sigma = {np.std(mreHist(vignets, pixelGrid))/np.sqrt(len(mreHist(vignets, pixelGrid)))}$', fontsize=30)

fig.savefig('pgfVsInterp.png')
fig2.savefig('logV_P_R.png')
fig3.savefig('MRE.png')
fig4.savefig('fftPk.png')
fig6.savefig('RelativeErrorHistogram.png')
fig6.savefig('RelativeErrorHistogram.pdf')
