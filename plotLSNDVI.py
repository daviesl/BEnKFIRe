import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import cm
from matplotlib import animation
from matplotlib import colors
from matplotlib.dates import datestr2num, num2date

dr = '/g/data/r78/lsd547/'
#fn = 'tas_landsat_dnbr.txt'
fn = 'waroona_landsat_ndvi_post_fire_2.txt'
#fn = 'waroona_landsat_ndvi_pre_fire_2.txt'
#fn = 'dnbr_SA_regional_20151201_31_H8_ALBERS.csv'

a = np.loadtxt(dr+fn)

fig, ax = plt.subplots()
cax = ax.imshow(a,vmin=-1,vmax=1)
cbar = fig.colorbar(cax, ticks=[-1,-0.5,-0.25, 0,0.25,0.5, 1], orientation='horizontal')
#cbar.ax.set_yticklabels(['-1','-0.5','-0.25','0','0.25','0.5', '1'])
plt.show()
