import gdal
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import animation
from matplotlib import colors
#import fmask
#dr='/g/data/rr5/satellite/obs/himawari8/FLDK/2016/01/06/0600/'
#fn='20160106060000-P1S-ABOM_OBS_B07-PRJ_GEOS141_2000-HIMAWARI8-AHI.nc'

i=0

dr='/g/data/r78/lsd547/H8/WA2/2016/01/'

#rast_arr = []
arr_arr = []
farr_arr = []
kfarr_arr = []
dnbr_arr = []

#b03rast_arr = []
b03arr_arr = []
#b04rast_arr = []
b04arr_arr = []
b14arr_arr = []
b11arr_arr = []
b15arr_arr = []
b13arr_arr = []
b07arr_arr = []

virnir_arr = []
ndvi_arr = []
singlerefl_arr = []
fog_arr = []
fog2_arr = []
cloud_arr = []
gross_arr = []
thin_arr = []

cols = 0
rows = 0
geotransform = None

days=(1,2,3,4,5,6,7,8,9,10)

for day in days:
	for hour in range(24):
		for tenminute in range(6):
			minute = tenminute * 10
			
			#fn = '20160106_1910_B07_Aus.tif'
			fn = '%02d/201601%02d_%02d%02d_NBR.tif'%(day,day,hour,minute)
			b3fn = '%02d/201601%02d_%02d%02d_B03_Aus.tif'%(day,day,hour,minute)
			b4fn = '%02d/201601%02d_%02d%02d_B04_Aus.tif'%(day,day,hour,minute)
			b11fn = '%02d/201601%02d_%02d%02d_B11_Aus.tif'%(day,day,hour,minute)
			b14fn = '%02d/201601%02d_%02d%02d_B14_Aus.tif'%(day,day,hour,minute)
			b15fn = '%02d/201601%02d_%02d%02d_B15_Aus.tif'%(day,day,hour,minute)
			b13fn = '%02d/201601%02d_%02d%02d_B13_Aus.tif'%(day,day,hour,minute)
			b7fn = '%02d/201601%02d_%02d%02d_B07_Aus.tif'%(day,day,hour,minute)
			print 'opening ' + dr + fn
			try:
				#B13
				r = gdal.Open(dr+b13fn)
				cols = r.RasterXSize
				rows = r.RasterYSize
				geotransform = r.GetGeoTransform()
				b13 = np.array(r.GetRasterBand(1).ReadAsArray())
				print 'loaded B13'
				#b14rast_arr.append(r )
				b13arr_arr.append( b13 )
				r = None
				#B11
				r = gdal.Open(dr+b11fn)
				b11 = np.array(r.GetRasterBand(1).ReadAsArray())
				print 'loaded B11'
				#b14rast_arr.append(r )
				b11arr_arr.append( b11 )
				r = None
				#B14
				r = gdal.Open(dr+b14fn)
				b14 = np.array(r.GetRasterBand(1).ReadAsArray())
				print 'loaded B14'
				#b14rast_arr.append(r )
				b14arr_arr.append( b14 )
				r = None
				#B15
				r = gdal.Open(dr+b15fn)
				b15 = np.array(r.GetRasterBand(1).ReadAsArray())
				print 'loaded B14'
				#b15rast_arr.append(r )
				b15arr_arr.append( b15 )
				r = None
				#B07
				r = gdal.Open(dr+b7fn)
				b7 = np.array(r.GetRasterBand(1).ReadAsArray())
				print 'loaded B07'
				#b07rast_arr.append(r )
				b07arr_arr.append( b7 )
				r = None
				#B03
				r = gdal.Open(dr+b3fn)
				b3 = np.array(r.GetRasterBand(1).ReadAsArray())
				print 'loaded B03'
				#b03rast_arr.append(r )
				b03arr_arr.append( b3 )
				r = None
				#B03
				r = gdal.Open(dr+b4fn)
				b4 = np.array(r.GetRasterBand(1).ReadAsArray())
				print 'loaded B04'
				#b04rast_arr.append(r )
				b04arr_arr.append( b4 )
				r = None
				# VIR / NIR ratio
				vn = np.divide(b4,b3)
				virnir_arr.append( vn )
				# Top Temperature
				# NDVI
				ndvi = np.divide(b4-b3,b4+b3)
				ndvi_arr.append(ndvi)
				#del r
				#NBR
				r = gdal.Open(dr+fn)
				a = np.array(r.GetRasterBand(1).ReadAsArray())
				print 'loaded nbr'
				#rast_arr.append(r )
				if 11*6 <= hour*6 + tenminute <= 21*6 + 2:
					a *= np.nan
				arr_arr.append( a )
				r = None
				i += 1
			except AttributeError, e:
				print e
				print "Unexpected error:", sys.exc_info()[0]
				print 'File for time %02d%02d does not exist'%(hour,minute)

B03_avg = np.ma.median(b03arr_arr[0:700],axis=0)
if not np.isfinite(B03_avg).all():
	print "B03 avg not finite"
	sys.exit()

B13_avg = np.ma.median(b13arr_arr[0:700],axis=0)
B14_avg = np.ma.median(b14arr_arr[0:700],axis=0)
B11_avg = np.ma.median(b11arr_arr[0:700],axis=0)
B15_avg = np.ma.median(b15arr_arr[0:700],axis=0)
B07_avg = np.ma.median(b07arr_arr[0:700],axis=0)
for i,b14 in enumerate(b14arr_arr):
	#srt = np.clip(np.divide(B03_avg,b3),0,1)
	gross = np.clip(np.divide(B13_avg,b13arr_arr[i]),0,10000)
	#singlerefl_arr.append(srt)
	gross_arr.append(gross)
	#c = np.exp(-np.power(virnir_arr[i]*srt,2))
	#c = np.exp(-np.power(virnir_arr[i] * gross,2))
	
	thin = np.divide(b14arr_arr[i]-b15arr_arr[i],B14_avg-B15_avg)
	#thin = np.clip(np.divide(b14arr_arr[i]-b15arr_arr[i],B14_avg-B15_avg),0,10000)
	#thin = np.clip(np.divide(b14arr_arr[i],b15arr_arr[i]),0,1000)
	thin_arr.append(thin)
	#fog = np.clip(np.divide(b14arr_arr[i]-b07arr_arr[i],B14_avg-B07_avg),0,10000)
	fog = np.divide(b14arr_arr[i]-b07arr_arr[i],B14_avg-B07_avg)
	#fog = np.clip(np.divide(b07arr_arr[i]-b14arr_arr[i],b07arr_arr[i]+b14arr_arr[i]),0,1000)
	fog_arr.append(fog)
	#fog2 = np.clip(np.divide(b14arr_arr[i]-b11arr_arr[i],B14_avg-B11_avg),0,10000)
	fog2 = np.divide(b14arr_arr[i]-b11arr_arr[i],B14_avg-B11_avg)
	#fog2 = np.clip(np.divide(b14arr_arr[i],b11arr_arr[i]),0,1000)
	fog2_arr.append(fog2)
	#c = np.exp(-np.power(virnir_arr[i] * thin,2))
	#c = np.exp(-virnir_arr[i] * thin * gross)
	#c = np.exp(-np.power(gross *virnir_arr[i],2))
	#c = np.exp(-np.minimum(thin,fog))
	#c = np.exp(-thin*fog*fog2*gross)
	#c = np.exp(-thin*fog*fog2)
	c = np.exp(-thin*fog2)
	
	cloud_arr.append(c)
	
	

if False:
	# the following code applies the median filter to the NBR
	for i,a in enumerate(arr_arr):
		if i > 7:
			fa = np.ma.median([a,arr_arr[i-1],arr_arr[i-2],arr_arr[i-3],arr_arr[i-4],arr_arr[i-5],arr_arr[i-6],arr_arr[i-7],arr_arr[i-8]],axis=0)
			farr_arr.append(fa)
		else:
			farr_arr.append(a)
	
	for i,a in enumerate(farr_arr):
		if i > 0:
			#b = np.clip(arr_arr[i-1]-a,0,2)
			#b /= np.amax(b)
			b = farr_arr[i-1]-a
			blowidx = b < 0.1
			b[blowidx] = -1
			dnbr_arr.append(b)
else:
	#median filter applied to dNBR
	#farr_arr = arr_arr
	for i,a in enumerate(arr_arr):
		if i > 1:
			fa = np.ma.median([a,arr_arr[i-1],arr_arr[i-2]],axis=0)
			farr_arr.append(fa)
		else:
			farr_arr.append(a)
	
	dim = arr_arr[0].shape
	xhat = np.zeros(dim)
	xhatminus = np.zeros(dim)
	P = np.ones(dim)
	Pminus = np.zeros(dim)
	K = np.zeros(dim)
	R = 0.2 ** 2 # estimate of measurement variance
	Rmin = 0.01 ** 2
	Q = 1e-5 # process variance
	print "dim " + str(dim)
	#for i,a in enumerate(farr_arr):
	for i,a in enumerate(arr_arr):
		if np.isfinite(virnir_arr[i]).all():
			#Rm = np.clip(R * np.exp(-np.power(virnir_arr[i]*1.5,2)),Rmin,R)
			Rm = np.clip(R * cloud_arr[i],Rmin,R)
		else:
			Rm = np.ones(dim) * R
		# time update
		if np.isfinite(xhat).all():
			xhatminus = np.copy(xhat)
		else:
			xhatminus = np.zeros(dim)
		Pminus = np.copy(P) + Q
		# measurement update
		K = Pminus / (Pminus+Rm)
		if np.isfinite(a).all():
			xhat = xhatminus + np.multiply( K, (a - xhatminus))
		P = np.multiply((1 - K),Pminus)
		kfarr_arr.append(xhat)
			
			
	for i,a in enumerate(kfarr_arr):
		if i > 0:
			b = kfarr_arr[i-1]-a
			#blowidx = b < -0.5
			#b[blowidx] = -1
			dnbr_arr.append(b)
		#	l = [a,arr_arr[i-1],arr_arr[i-2],arr_arr[i-3],arr_arr[i-4],arr_arr[i-5],arr_arr[i-6],arr_arr[i-7],arr_arr[i-8]]
		#	#l = [a,arr_arr[i-1],arr_arr[i-2],arr_arr[i-3],arr_arr[i-4]]
		#	dl = [l[i]-l[i+1] for i in xrange(len(l)-1)]
		#	print "computing median"
		#	b = np.ma.median(dl,axis=0)
		#	blowidx = b < -0.5
		#	b[blowidx] = -1
		#	dnbr_arr.append(b)


#raster = rast_arr[0]
#geotransform = raster.GetGeoTransform()
originX = geotransform[0]
originY = geotransform[3]
pixelWidth = geotransform[1]
pixelHeight = geotransform[5]
#xOffset = int((x - originX)/pixelWidth)
#yOffset = int((y - originY)/pixelHeight)

#cols = raster.RasterXSize
#rows = raster.RasterYSize
print 'Origin: (%f, %f)'%(originX,originY)

fig = plt.figure(figsize=(50/3,12.5))
gs = gridspec.GridSpec(4,2, width_ratios=[1,1])
axNBR = fig.add_subplot(gs[0,0])
axMNBR = fig.add_subplot(gs[0,1])
axKNBR = fig.add_subplot(gs[1,0])
axDNBR = fig.add_subplot(gs[1,1])
axVIR = fig.add_subplot(gs[2,0])
axCLD = fig.add_subplot(gs[2,1])
axFOG = fig.add_subplot(gs[3,0])
axFOG2 = fig.add_subplot(gs[3,1])
imNBR = axNBR.imshow(farr_arr[0],cmap='Greys_r',interpolation='none', vmin=-1, vmax=1, extent=(originX,originX+pixelWidth*cols,originY+pixelHeight*rows,originY))
imMNBR = axMNBR.imshow(farr_arr[0],cmap='Greys_r',interpolation='none', vmin=-1, vmax=1, extent=(originX,originX+pixelWidth*cols,originY+pixelHeight*rows,originY))
imKNBR = axKNBR.imshow(kfarr_arr[0],cmap='Greys_r',interpolation='none', vmin=-1, vmax=1, extent=(originX,originX+pixelWidth*cols,originY+pixelHeight*rows,originY))
imDNBR = axDNBR.imshow(dnbr_arr[0],cmap='jet',interpolation='none', vmin=-0.5, vmax=0.5, extent=(originX,originX+pixelWidth*cols,originY+pixelHeight*rows,originY))
imVIR = axVIR.imshow(thin_arr[0],cmap='Greys_r',interpolation='none', vmin=0, vmax=10, extent=(originX,originX+pixelWidth*cols,originY+pixelHeight*rows,originY))
imCLD = axCLD.imshow(cloud_arr[0],cmap='Greys_r',interpolation='none', vmin=0, vmax=1, extent=(originX,originX+pixelWidth*cols,originY+pixelHeight*rows,originY))
imFOG = axFOG.imshow(fog_arr[0],cmap='Greys_r',interpolation='none', vmin=0, vmax=10, extent=(originX,originX+pixelWidth*cols,originY+pixelHeight*rows,originY))
imFOG2 = axFOG2.imshow(fog2_arr[0],cmap='Greys_r',interpolation='none', vmin=0, vmax=10, extent=(originX,originX+pixelWidth*cols,originY+pixelHeight*rows,originY))
#axNBR.colorbar()

print 'Initted figure'

# The animation function: called to produce a frame for each generation.
def animate(i):
	print 'setting im'
	imDNBR.set_data(animate.dnbr_arr[animate.Iter])
	imNBR.set_data(animate.nbr_arr[animate.Iter])
	imMNBR.set_data(animate.mnbr_arr[animate.Iter])
	imKNBR.set_data(animate.knbr_arr[animate.Iter])
	imVIR.set_data(animate.thin_arr[animate.Iter])
	imFOG.set_data(animate.fog_arr[animate.Iter])
	imFOG2.set_data(animate.fog2_arr[animate.Iter])
	imCLD.set_data(animate.cloud_arr[animate.Iter])
	#ttl.set_text('Iteration: %d (1 simulated second per iteration)'%(animate.Iter))
	if animate.Iter < i-1:
		animate.Iter += 1
#animate.raster = rast_arr
#animate.arr = arr_arr
animate.dnbr_arr = dnbr_arr
animate.knbr_arr = kfarr_arr
animate.mnbr_arr = farr_arr
animate.nbr_arr = arr_arr
animate.thin_arr = thin_arr #virnir_arr
animate.fog_arr = fog_arr
animate.fog2_arr = fog2_arr
animate.cloud_arr = cloud_arr #thin_arr #singlerefl_arr
animate.Iter = 0

print 'Set animate'
print len(animate.dnbr_arr)
print i

interval = 200
# Bind our grid to the identifier X in the animate function's namespace.
#Writer = animation.writers['ffmpeg']
#writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
anim = animation.FuncAnimation(fig, animate, interval=interval, repeat=False, frames=len(dnbr_arr))
#anim.save('dnbr_filtered.mp4',writer=writer)
plt.show()
