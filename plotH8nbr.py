import gdal
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import animation
from matplotlib import colors
#dr='/g/data/rr5/satellite/obs/himawari8/FLDK/2016/01/06/0600/'
#fn='20160106060000-P1S-ABOM_OBS_B07-PRJ_GEOS141_2000-HIMAWARI8-AHI.nc'

i=0

dr='/g/data/r78/lsd547/H8/WA/2016/01/'

rast_arr = []
arr_arr = []

days=(6,7,8,9,10)

for day in days:
	for hour in range(24):
		for tenminute in range(6):
			minute = tenminute * 10
			
			#fn = '20160106_1910_B07_Aus.tif'
			fn = '%02d/201601%02d_%02d%02d_NBR.tif'%(day,day,hour,minute)
			print 'opening ' + dr + fn
			try:
				r = gdal.Open(dr+fn)
				a = np.array(r.GetRasterBand(1).ReadAsArray())
				print 'loaded'
				rast_arr.append(r )
				arr_arr.append( a )
				i += 1
			except AttributeError, e:
				print e
				print "Unexpected error:", sys.exc_info()[0]
				print 'File for time %02d%02d does not exist'%(hour,minute)

dnbr_arr = []
for i,a in enumerate(arr_arr):
	if i > 0:
		b = np.clip(arr_arr[i-1]-a,0,2)
		b /= np.amax(b)
		dnbr_arr.append(b)

raster = rast_arr[0]
geotransform = raster.GetGeoTransform()
originX = geotransform[0]
originY = geotransform[3]
pixelWidth = geotransform[1]
pixelHeight = geotransform[5]
#xOffset = int((x - originX)/pixelWidth)
#yOffset = int((y - originY)/pixelHeight)
cols = raster.RasterXSize
rows = raster.RasterYSize

print 'Origin: (%f, %f)'%(originX,originY)

fig = plt.figure()
gs = gridspec.GridSpec(1,2, width_ratios=[1,1])
axNBR = fig.add_subplot(gs[0,0])
axDNBR = fig.add_subplot(gs[0,1])
imDNBR = axDNBR.imshow(arr_arr[0],cmap='jet',interpolation='none', vmin=0, vmax=1, extent=(originX,originX+pixelWidth*cols,originY+pixelHeight*rows,originY))
imNBR = axNBR.imshow(arr_arr[0],cmap='jet',interpolation='none', vmin=0, vmax=1, extent=(originX,originX+pixelWidth*cols,originY+pixelHeight*rows,originY))
#axNBR.colorbar()

print 'Initted figure'

# The animation function: called to produce a frame for each generation.
def animate(i):
	print 'setting im'
	imDNBR.set_data(animate.dnbr_arr[animate.Iter])
	imNBR.set_data(animate.nbr_arr[animate.Iter])
	#ttl.set_text('Iteration: %d (1 simulated second per iteration)'%(animate.Iter))
	if animate.Iter < i-1:
		animate.Iter += 1
animate.raster = rast_arr
#animate.arr = arr_arr
animate.dnbr_arr = dnbr_arr
animate.nbr_arr = arr_arr
animate.Iter = 0

print 'Set animate'
print len(animate.dnbr_arr)
print i

interval = 100
# Bind our grid to the identifier X in the animate function's namespace.
#Writer = animation.writers['ffmpeg']
#writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
anim = animation.FuncAnimation(fig, animate, interval=interval)
#anim.save('dnbr.mp4',writer=writer)
plt.show()
