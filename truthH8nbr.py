import gdal
import sys
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import animation
from matplotlib import colors
from matplotlib.dates import datestr2num, num2date
import bisect

i=0

#WA
#dr='/g/data/r78/lsd547/H8/WA2/2016/01/'
#hs_h8_file='WA_20160101_30_H8_ALBERS.csv'
#hs_MODIS_file='WA_jan2016_MODIS_VIIRS_ALBERS.csv'

#TAS
dr='/g/data/r78/lsd547/H8/TAS/2016/01/'
hs_MODIS_file='TAS_20160113_30_MODIS_ALBERS.csv'
hs_h8_file='TAS_Jan01_21_H8_ALBERS_dt.csv'

#rast_arr = []
arr_arr = []
mndviavg_arr = []
farr_arr = []
kfndvi_arr = []
dndvi_arr = []

kfnbr_arr = []
dnbr_arr = []
nbr_arr = []


#b03rast_arr = []
b03arr_arr = []
#b04rast_arr = []
b04arr_arr = []
b06arr_arr = []
b14arr_arr = []
b11arr_arr = []
b15arr_arr = []
b13arr_arr = []
b07arr_arr = []
epoch_arr = []

virnir_arr = []
ndvi_arr = []
singlerefl_arr = []
fog_arr = []
fog2_arr = []
cloud_arr = []
KGain_arr = []
gross_arr = []
thin_arr = []

hs_show_arr = []

cols = 0
rows = 0
geotransform = None

#days=(1,2)
#WA
#days=(1,2,3,4,5,6,7,8,9,10)
#TAS
days=(10,11,12,13,14,15,16,17,18)

for day in days:
	for hour in range(24):
		for tenminute in range(6):
			minute = tenminute * 10
			
			#fn = '20160106_1910_B07_Aus.tif'
			fn = '%02d/201601%02d_%02d%02d_NBR.tif'%(day,day,hour,minute)
			b3fn = '%02d/201601%02d_%02d%02d_B03_Aus.tif'%(day,day,hour,minute)
			b4fn = '%02d/201601%02d_%02d%02d_B04_Aus.tif'%(day,day,hour,minute)
			b6fn = '%02d/201601%02d_%02d%02d_B06_Aus.tif'%(day,day,hour,minute)
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
				#B04
				r = gdal.Open(dr+b4fn)
				b4 = np.array(r.GetRasterBand(1).ReadAsArray())
				print 'loaded B04'
				#b04rast_arr.append(r )
				b04arr_arr.append( b4 )
				r = None
				#B06
				r = gdal.Open(dr+b6fn)
				b6 = np.array(r.GetRasterBand(1).ReadAsArray())
				print 'loaded B06'
				#b06rast_arr.append(r )
				b06arr_arr.append( b6 )
				r = None
				# VIR / NIR ratio
				vn = np.divide(b4,b3)
				virnir_arr.append( vn )
				# Top Temperature
				# NDVI
				#ndvi = np.divide(b4-b3,b4+b3)
				#ndvi_arr.append(ndvi)
				#del r
				#NBR
				r = gdal.Open(dr+fn)
				a = np.array(r.GetRasterBand(1).ReadAsArray())
				print 'loaded nbr'
				#rast_arr.append(r )
				if 11*6 <= hour*6 + tenminute <= 21*6 + 2:
					#a *= np.nan
					a[0,0] = np.nan
				arr_arr.append( a )
				r = None
				epoch_arr.append(datestr2num('2016-01-%02d %02d:%02d:00'%(day,hour,minute)))
				hs_show_arr.append([])
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
B04_avg = np.ma.median(b04arr_arr[0:700],axis=0)
B06_avg = np.ma.median(b06arr_arr[0:700],axis=0)
B13_std = np.ma.std(b13arr_arr[0:700],axis=0)
B14_avg = np.ma.median(b14arr_arr[0:700],axis=0)
B11_avg = np.ma.median(b11arr_arr[0:700],axis=0)
B15_avg = np.ma.median(b15arr_arr[0:700],axis=0)
B07_avg = np.ma.median(b07arr_arr[0:700],axis=0)
for i,b14 in enumerate(b14arr_arr):
	#srt = np.clip(np.divide(B03_avg,b3),0,1)
	#singlerefl_arr.append(srt)

	#gross = np.clip(np.divide(B13_avg,b13arr_arr[i]),0,10000)
	#gross = np.clip(np.divide(b13arr_arr[i] - B13_avg,B13_avg),0,1)
	#gross = np.clip(np.divide(b13arr_arr[i] - B13_avg,B13_std),0,1)

	#gross = np.clip(b13arr_arr[i] - B13_avg,0,10000)
	gross = b13arr_arr[i] - B13_avg

	gross_arr.append(gross)
	#c = np.exp(-np.power(virnir_arr[i]*srt,2))
	#c = np.exp(-np.power(virnir_arr[i] * gross,2))
	
	#thin = np.clip(np.divide(b14arr_arr[i]-b15arr_arr[i],B14_avg-B15_avg),0,10000)
	thin = np.divide(b14arr_arr[i]-b15arr_arr[i],B14_avg-B15_avg)

	#thin = np.divide(b14arr_arr[i]-b15arr_arr[i],B14_avg-B15_avg)
	#thin = np.clip(np.divide(b14arr_arr[i],b15arr_arr[i]),0,1000)
	thin_arr.append(thin)

	#fog = np.clip(np.divide(b14arr_arr[i]-b07arr_arr[i],B14_avg-B07_avg),0,10000)
	fog = np.divide(b14arr_arr[i]-b07arr_arr[i],B14_avg-B07_avg)

	#fog = np.divide(b14arr_arr[i]-b07arr_arr[i],B14_avg-B07_avg)
	#fog = np.clip(np.divide(b07arr_arr[i]-b14arr_arr[i],b07arr_arr[i]+b14arr_arr[i]),0,1000)
	fog_arr.append(fog)

	#fog2 = np.clip(np.divide(b14arr_arr[i]-b11arr_arr[i],B14_avg-B11_avg),0,10000)
	fog2 = np.divide(b14arr_arr[i]-b11arr_arr[i],B14_avg-B11_avg)

	#fog2 = np.divide(b14arr_arr[i]-b11arr_arr[i],B14_avg-B11_avg)
	#fog2 = np.clip(np.divide(b14arr_arr[i],b11arr_arr[i]),0,1000)
	fog2_arr.append(fog2)

	#c = np.exp(-np.power(virnir_arr[i] * thin,2))
	#c = np.exp(-virnir_arr[i] * thin * gross)
	#c = np.exp(-np.power(gross *virnir_arr[i],2))
	#c = np.exp(-np.minimum(thin,fog))
	#c = np.exp(-thin*fog*fog2*gross)
	#c = np.exp(-thin*fog*fog2)

	#c = np.exp(-thin*fog2)
	#c = np.exp(-(0.5*(thin + fog2)))
	#c = np.exp(-(0.33*(gross + thin + fog2)))
	#cloud_arr.append(c)

	#mndviavg = np.divide((b04arr_arr[i] - b07arr_arr[i]),(B04_avg - B07_avg))
	#mndviavg = np.divide(b04arr_arr[i]- b07arr_arr[i] - (B04_avg - B07_avg),b04arr_arr[i]+ b07arr_arr[i] + (B04_avg - B07_avg))
	mndviavg = np.divide(b04arr_arr[i]- b07arr_arr[i], (B04_avg + B07_avg))
	#if not np.isfinite(arr_arr[i]).all():
	#	mndviavg[0,0] = np.nan
	#mndviavg_arr.append(mndviavg)
	ndvi_arr.append(mndviavg)

	mnbravg = np.divide(b04arr_arr[i]- b06arr_arr[i], (B04_avg + B06_avg))
	nbr_arr.append(mnbravg)
	
emin = 2.7 * 3
gross_scale = emin/30
thin_scale = emin/6
fog_scale = emin/35
fog2_scale = emin/3
for i,g in enumerate(gross_arr):
	c = np.exp(-(0.33*(gross_scale * gross_arr[i] + thin_scale * thin_arr[i] + fog2_scale * fog2_arr[i])))
	cloud_arr.append(c)

	
#arr_arr = mndviavg_arr
	

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
			#blowidx = b < 0.1
			#b[blowidx] = -1
			dndvi_arr.append(b)
else:
	#median filter applied to dNBR
	for i,a in enumerate(ndvi_arr):
		if i > 1:
			fa = np.ma.median([a,ndvi_arr[i-1],ndvi_arr[i-2]],axis=0)
			farr_arr.append(fa)
		else:
			farr_arr.append(a)
	
	dim = ndvi_arr[0].shape
	xhat = np.zeros(dim)
	xhatminus = np.zeros(dim)
	P = np.ones(dim)
	Pminus = np.zeros(dim)
	K = np.zeros(dim)
	R = 0.2 ** 2 # estimate of measurement variance
	Rmin = 0.005 ** 2
	Q = 1e-5 # process variance
	print "dim " + str(dim)
	for i,a in enumerate(ndvi_arr):
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
		if np.isfinite(a).all():
			yhat = a - xhatminus
			# reweight by yhat vs median
			K = np.divide(Pminus, (Pminus+Rm))
			xhat = xhatminus + np.multiply( K, yhat)
			P = np.multiply((1 - K),Pminus)
		kfndvi_arr.append(xhat)
		KGain_arr.append(Rm)
			
	edif = 11			
	for j in range(edif):
		dndvi_arr.append(np.zeros(dim))
		
	for i,a in enumerate(kfndvi_arr):
		if i > edif:
			#b = np.median([kfndvi_arr[i-edif-1],kfndvi_arr[i-edif],kfndvi_arr[i-edif+1]],axis=0)-np.median([kfndvi_arr[i-2],kfndvi_arr[i-1],a],axis=0)
			b = np.median([kfndvi_arr[i-edif-1],kfndvi_arr[i-edif],kfndvi_arr[i-edif+1]],axis=0)-a
			#blowidx = b < 0.3
			#b[blowidx] = -1
			dndvi_arr.append(b)
		#else:
		#	dndvi_arr.append(np.zeros(dim))
	# NBR filtering
	dim = nbr_arr[0].shape
	xhat = np.zeros(dim)
	xhatminus = np.zeros(dim)
	P = np.ones(dim)
	Pminus = np.zeros(dim)
	K = np.zeros(dim)
	R = 0.2 ** 2 # estimate of measurement variance
	Rmin = 0.005 ** 2
	Q = 1e-5 # process variance
	print "dim " + str(dim)
	for i,a in enumerate(nbr_arr):
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
		if np.isfinite(a).all():
			yhat = a - xhatminus
			# reweight by yhat vs median
			K = np.divide(Pminus, (Pminus+Rm))
			xhat = xhatminus + np.multiply( K, yhat)
			P = np.multiply((1 - K),Pminus)
		kfnbr_arr.append(xhat)
		KGain_arr.append(Rm)
			
	edif = 11			
	for j in range(edif):
		dnbr_arr.append(np.zeros(dim))
		
	for i,a in enumerate(kfnbr_arr):
		if i > edif:
			#b = np.median([kfnbr_arr[i-edif-1],kfnbr_arr[i-edif],kfnbr_arr[i-edif+1]],axis=0)-np.median([kfnbr_arr[i-2],kfnbr_arr[i-1],a],axis=0)
			b = np.median([kfnbr_arr[i-edif-1],kfnbr_arr[i-edif],kfnbr_arr[i-edif+1]],axis=0)-a
			#blowidx = b < 0.3
			#b[blowidx] = -1
			dnbr_arr.append(b)
		#else:
		#	dnbr_arr.append(np.zeros(dim))


#raster = rast_arr[0]
#geotransform = raster.GetGeoTransform()
originX = geotransform[0]
originY = geotransform[3]
pixelWidth = geotransform[1]
pixelHeight = geotransform[5]
#if pixelHeight < 0:
#	pixelHeight = -pixelHeight
#	originY -= pixelHeight * rows
#xOffset = int((x - originX)/pixelWidth)
#yOffset = int((y - originY)/pixelHeight)

#cols = raster.RasterXSize
#rows = raster.RasterYSize
print 'Origin: (%f, %f) Pixel: (%f, %f)'%(originX,originY,pixelWidth,pixelHeight)

def sv2num(st):
	#print st
	val = {
		'2011-061A':0.0, #VIIRS
		'2002-022A':1.0, # MODIS
		'1999-068A':2.0, # MODIS
		'2009-005A':3.0, # AVHRR
		}[st]
	#print val
	return val

def utc2num(st):
	#convert dates. For now return zero
	return datestr2num(st)
	#return 0.0

def floatOrBlank(st):
	if len(st)==0:
		return -9999
	else:
		return float(st)

#Load hotspots
#MODIS/VIIRS hotspots

# WA
#hs_MODIS = np.loadtxt(hs_MODIS_file,skiprows=1,delimiter=',',usecols=(0,1,19,20,13,15,26,27,28,29,30,31,32), converters={15:sv2num,19:utc2num,20:utc2num,28:floatOrBlank,29:utc2num,30:floatOrBlank,31:floatOrBlank}) # X, Y, start_dt, end_dt, id, satellite_, lat, lon, temp_K, datetime, power, confidence, age_hours 

modis_file_cols = (0,1,8,9,2,4,15,16,17,18,19,20,21) # TAS
hs_MODIS = np.loadtxt(hs_MODIS_file,skiprows=1,delimiter=',',usecols=modis_file_cols, converters={4:sv2num,8:utc2num,9:utc2num,17:floatOrBlank,18:utc2num,19:floatOrBlank,20:floatOrBlank}) # X, Y, start_dt, end_dt, id, satellite_, lat, lon, temp_K, datetime, power, confidence, age_hours 

hs_MODIS[hs_MODIS[:,2].argsort()] #3rd column is start_dt
hs_MODIS_radius = 0.5 * 500.0 / 1000.0
(hs_rows,hs_cols) = hs_MODIS.shape
hs_MODIS = np.concatenate((hs_MODIS,np.zeros((hs_rows,18))),axis=1)
synth_MODIS = []
for i in xrange(hs_rows):
	# Find dNBR by epoch and pixel
	# float search - substract tiny number and bisect left to get insertion index
	row = 1.0 * (hs_MODIS[i,1] - originY) / pixelHeight
	col = 1.0 * (hs_MODIS[i,0] - originX) / pixelWidth
	if 0 <= row < rows and 0 <= col <= cols:
		nearest_epoch_idx = bisect.bisect_left(epoch_arr,hs_MODIS[i,9]-0.00001)
		#print "Nearest epoch idx = " + str(nearest_epoch_idx)
		tenmin = math.fmod(hs_MODIS[i,9],1.0) * 144
		if nearest_epoch_idx >= len(dndvi_arr) or nearest_epoch_idx < 0:
			hs_MODIS[i,hs_cols + 0] = 0
		else:
			if 11*6 <= tenmin <= 21*6 + 2:
				hs_MODIS[i,hs_cols + 14] = 0 # daytime flag
			else:
				hs_MODIS[i,hs_cols + 14] = 1 # daytime flag
			hs_show_arr[nearest_epoch_idx].append((hs_MODIS[i,0],hs_MODIS[i,1],19)) #[0.0,0.0,1.0,0.5])) # 'b'
			hs_MODIS[i,hs_cols + 0] = 1
			#epochdelta = abs(epoch_arr[nearest_epoch_idx] - hs_MODIS[i,9])
			epochdelta = epoch_arr[nearest_epoch_idx] - hs_MODIS[i,9]
			hs_MODIS[i,hs_cols + 15] = epochdelta # daytime flag
			#if epochdelta < 0.5/144: # five minute difference
			d = dndvi_arr[nearest_epoch_idx]
			hs_MODIS[i,hs_cols + 16] = row
			hs_MODIS[i,hs_cols + 17] = col
			#c_val = cloud_arr[nearest_epoch_idx][int(row),int(col)]
			dval = dndvi_arr[nearest_epoch_idx][int(row),int(col)]
			ndvival = kfndvi_arr[nearest_epoch_idx][int(row),int(col)]
			rawndvival = ndvi_arr[nearest_epoch_idx][int(row),int(col)]
			nbrval = kfnbr_arr[nearest_epoch_idx][int(row),int(col)]
			dnbrval = dnbr_arr[nearest_epoch_idx][int(row),int(col)]
			rawnbrval = nbr_arr[nearest_epoch_idx][int(row),int(col)]
			b7val = b07arr_arr[nearest_epoch_idx][int(row),int(col)]
			b6val = b06arr_arr[nearest_epoch_idx][int(row),int(col)]
			b14val = b14arr_arr[nearest_epoch_idx][int(row),int(col)]
			gross_val = gross_arr[nearest_epoch_idx][int(row),int(col)]
			thin_val = thin_arr[nearest_epoch_idx][int(row),int(col)]
			fog_val = fog_arr[nearest_epoch_idx][int(row),int(col)]
			fog2_val = fog2_arr[nearest_epoch_idx][int(row),int(col)]
			hs_MODIS[i,hs_cols + 1] = dval
			hs_MODIS[i,hs_cols + 2] = ndvival
			hs_MODIS[i,hs_cols + 3] = rawndvival
			hs_MODIS[i,hs_cols + 4] = dnbrval
			hs_MODIS[i,hs_cols + 5] = nbrval
			hs_MODIS[i,hs_cols + 6] = rawnbrval
			hs_MODIS[i,hs_cols + 7] = b6val
			hs_MODIS[i,hs_cols + 8] = b7val
			hs_MODIS[i,hs_cols + 9] = b14val
			hs_MODIS[i,hs_cols + 10] = gross_val
			hs_MODIS[i,hs_cols + 11] = thin_val
			hs_MODIS[i,hs_cols + 12] = fog_val
			hs_MODIS[i,hs_cols + 13] = fog2_val
			# Now sample dnbr between Jan 01 and Jan 04 2016 daytime only
			rand_epoch_idx = 0
			while True:
				#tenmin = np.random.randint(0,14*6-3)
				#if tenmin >= 11*6:
				#	tenmin += 10*6+3
				hour = int(tenmin / 6)
				minute = int(tenmin % 6) * 10
				rand_epoch = utc2num('2016-01-%02d %02d:%02d:00'%(np.random.randint(1,5),hour,minute))
				rand_epoch_idx = bisect.bisect_left(epoch_arr,rand_epoch - 0.00001)
				if rand_epoch_idx < len(dndvi_arr):
					#rand_epoch_idx = np.random.randint(0,568) # 4 days
					dval = dndvi_arr[rand_epoch_idx][int(row),int(col)]
					ndvival = kfndvi_arr[rand_epoch_idx][int(row),int(col)]
					rawndvival = ndvi_arr[rand_epoch_idx][int(row),int(col)]
					nbrval = kfnbr_arr[rand_epoch_idx][int(row),int(col)]
					dnbrval = dnbr_arr[rand_epoch_idx][int(row),int(col)]
					rawnbrval = nbr_arr[rand_epoch_idx][int(row),int(col)]
					b7val = b07arr_arr[rand_epoch_idx][int(row),int(col)]
					b6val = b06arr_arr[rand_epoch_idx][int(row),int(col)]
					b14val = b14arr_arr[rand_epoch_idx][int(row),int(col)]
					gross_val = gross_arr[rand_epoch_idx][int(row),int(col)]
					thin_val = thin_arr[rand_epoch_idx][int(row),int(col)]
					fog_val = fog_arr[rand_epoch_idx][int(row),int(col)]
					fog2_val = fog2_arr[rand_epoch_idx][int(row),int(col)]
					# Now sample dnbr between Jan 01 and Jan 04 2016
					synth_MODIS.append([hs_MODIS[i,0],hs_MODIS[i,1],rand_epoch,dval,ndvival,rawndvival,dnbrval,nbrval,rawnbrval,b6val,b7val,b14val,gross_val,thin_val,fog_val,fog2_val,row,col])
					hs_show_arr[rand_epoch_idx].append((hs_MODIS[i,0],hs_MODIS[i,1],0)) #[0.0,0.0,1.0,0.5])) # 'b'
					break
	else:
		hs_MODIS[i,hs_cols + 0] = 0
		hs_MODIS[i,hs_cols + 1] = -9999
		continue

np.savetxt('dnbr_'+hs_MODIS_file,hs_MODIS,delimiter=',')
np.savetxt('dnbr_synth_'+hs_MODIS_file,np.array(synth_MODIS),delimiter=',')
	
# H8 Hotspots
# X,Y,Datetime,Lat,Lon,FRP(mw),Firesize(km^2),temp(k),Category,Description
hs_h8 = np.loadtxt(hs_h8_file,skiprows=1,delimiter=',',usecols=(0,1,2,3,4,5,6,7,8),converters={2:datestr2num})
hs_h8[hs_h8[:,2].argsort()] #3rd column is utc time
(hs_rows,hs_cols) = hs_h8.shape
hs_h8 = np.concatenate((hs_h8,np.zeros((hs_rows,18))),axis=1)
#hs_h8_epochs = [r[2] for r in hs_h8]
hs_h8_radius = 0.5 * 2000.0 / 1000.0
for i in xrange(hs_rows):
	# Find dNBR by epoch and pixel
	# float search - substract tiny number and bisect left to get insertion index
	nearest_epoch_idx = bisect.bisect_left(epoch_arr,hs_h8[i,2]-0.00001)
	#print "Nearest epoch idx = " + str(nearest_epoch_idx)
	if nearest_epoch_idx >= len(epoch_arr)-5 or nearest_epoch_idx < 0:
		hs_h8[i,hs_cols + 0] = 0
		continue
	else:
		hs_h8[i,hs_cols + 0] = 1
	hs_show_arr[nearest_epoch_idx].append((hs_h8[i,0],hs_h8[i,1],10)) #[1.0,1.0,0.0,0.5])) # 'y'
	#epochdelta = abs(epoch_arr[nearest_epoch_idx] - hs_h8[i,2])
	#if epochdelta < 0.5/144: # five minute difference
	epochdelta =  hs_h8[i,2] - epoch_arr[nearest_epoch_idx]
	row = 1.0 * (hs_h8[i,1] - originY) / pixelHeight
	col = 1.0 * (hs_h8[i,0] - originX) / pixelWidth
	hs_h8[i,hs_cols + 17] = row
	hs_h8[i,hs_cols + 16] = col
	hs_h8[i,hs_cols + 15] = epochdelta
	tenmin = math.fmod(hs_h8[i,2],1.0) * 144
	if 11*6 <= tenmin <= 21*6 + 2:
		hs_h8[i,hs_cols + 14] = 0 # daytime flag
	else:
		hs_h8[i,hs_cols + 14] = 1 # daytime flag
	if 0 <= row < rows and 0 <= col <= cols:
		#d = dndvi_arr[nearest_epoch_idx]
		#dval = d[int(row),int(col)]
		#nbr = kfndvi_arr[nearest_epoch_idx]
		#ndvival = nbr[int(row),int(col)]
		#b7_ = b07arr_arr[nearest_epoch_idx]
		#b7val = b7_[int(row),int(col)]
		#b14_ = b14arr_arr[nearest_epoch_idx]
		#c_val = cloud_arr[nearest_epoch_idx][int(row),int(col)]
		#hs_h8[i,hs_cols + 1] = dval
		#hs_h8[i,hs_cols + 2] = ndvival
		#hs_h8[i,hs_cols + 3] = b7val
		#hs_h8[i,hs_cols + 4] = b14val
		#hs_h8[i,hs_cols + 5] = c_val
		dval = dndvi_arr[nearest_epoch_idx][int(row),int(col)]
		ndvival = kfndvi_arr[nearest_epoch_idx][int(row),int(col)]
		rawndvival = ndvi_arr[nearest_epoch_idx][int(row),int(col)]
		nbrval = kfnbr_arr[nearest_epoch_idx][int(row),int(col)]
		dnbrval = dnbr_arr[nearest_epoch_idx][int(row),int(col)]
		rawnbrval = nbr_arr[nearest_epoch_idx][int(row),int(col)]
		b7val = b07arr_arr[nearest_epoch_idx][int(row),int(col)]
		b6val = b06arr_arr[nearest_epoch_idx][int(row),int(col)]
		b14val = b14arr_arr[nearest_epoch_idx][int(row),int(col)]
		gross_val = gross_arr[nearest_epoch_idx][int(row),int(col)]
		thin_val = thin_arr[nearest_epoch_idx][int(row),int(col)]
		fog_val = fog_arr[nearest_epoch_idx][int(row),int(col)]
		fog2_val = fog2_arr[nearest_epoch_idx][int(row),int(col)]
		hs_h8[i,hs_cols + 1] = dval
		hs_h8[i,hs_cols + 2] = ndvival
		hs_h8[i,hs_cols + 3] = rawndvival
		hs_h8[i,hs_cols + 4] = dnbrval
		hs_h8[i,hs_cols + 5] = nbrval
		hs_h8[i,hs_cols + 6] = rawnbrval
		hs_h8[i,hs_cols + 7] = b6val
		hs_h8[i,hs_cols + 8] = b7val
		hs_h8[i,hs_cols + 9] = b14val
		hs_h8[i,hs_cols + 10] = gross_val
		hs_h8[i,hs_cols + 11] = thin_val
		hs_h8[i,hs_cols + 12] = fog_val
		hs_h8[i,hs_cols + 13] = fog2_val
	else:
		hs_h8[i,hs_cols + 0] = 0
		#hs_h8[i,10] = -9999
		continue

np.savetxt('dnbr_'+hs_h8_file,hs_h8,delimiter=',')
			
			
	
	



fig = plt.figure(figsize=(50/3,12.5))
gs = gridspec.GridSpec(3,3) #, width_ratios=[1,1])
#axMNBR = fig.add_subplot(gs[0,1])
axNBR = fig.add_subplot(gs[0,0])
axKNBR = fig.add_subplot(gs[0,1])
axDNBR = fig.add_subplot(gs[0,2])
axNDVI = fig.add_subplot(gs[1,0])
axKNDVI = fig.add_subplot(gs[1,1])
axDNDVI = fig.add_subplot(gs[1,2])
axVIR = fig.add_subplot(gs[2,0])
axFOG = fig.add_subplot(gs[2,1])
axFOG2 = fig.add_subplot(gs[2,2])
#axCLD = fig.add_subplot(gs[3,:])
#imMNBR = axMNBR.imshow(farr_arr[0],cmap='Greys_r',interpolation='none', vmin=-2, vmax=0, extent=(originX,originX+pixelWidth*cols,originY+pixelHeight*rows,originY))
imNBR = axNBR.imshow(nbr_arr[0],cmap='Greys_r',interpolation='none', vmin=-2, vmax=0, extent=(originX,originX+pixelWidth*cols,originY+pixelHeight*rows,originY))
imKNBR = axKNBR.imshow(kfnbr_arr[0],cmap='Greys_r',interpolation='none', vmin=-2, vmax=0, extent=(originX,originX+pixelWidth*cols,originY+pixelHeight*rows,originY))
imDNBR = axDNBR.imshow(dnbr_arr[0],cmap='jet',interpolation='none', vmin=-0.5, vmax=0.5, extent=(originX,originX+pixelWidth*cols,originY+pixelHeight*rows,originY))
imNDVI = axNDVI.imshow(ndvi_arr[0],cmap='Greys_r',interpolation='none', vmin=-2, vmax=0, extent=(originX,originX+pixelWidth*cols,originY+pixelHeight*rows,originY))
imKNDVI = axKNDVI.imshow(kfndvi_arr[0],cmap='Greys_r',interpolation='none', vmin=-2, vmax=0, extent=(originX,originX+pixelWidth*cols,originY+pixelHeight*rows,originY))
imDNDVI = axDNDVI.imshow(dndvi_arr[0],cmap='jet',interpolation='none', vmin=-0.5, vmax=0.5, extent=(originX,originX+pixelWidth*cols,originY+pixelHeight*rows,originY))
axDNBR.autoscale(False)
#scatterDNBR = axDNBR.scatter(x=[x[0] for x in hs_show_arr[0]],y=[x[1] for x in hs_show_arr[0]],c=[[x[2]] for x in hs_show_arr[0]])
cm = plt.cm.get_cmap('RdYlBu')
scatterDNBR = axDNBR.scatter(x=[0],y=[0],c=[0],cmap=cm,vmin=0,vmax=20)
imVIR = axVIR.imshow(thin_arr[0],cmap='Greys_r',interpolation='none', vmin=0, vmax=5, extent=(originX,originX+pixelWidth*cols,originY+pixelHeight*rows,originY))
#imCLD = axCLD.imshow(cloud_arr[0],cmap='Greys_r',interpolation='none', vmin=0, vmax=0.1, extent=(originX,originX+pixelWidth*cols,originY+pixelHeight*rows,originY))
imFOG = axFOG.imshow(gross_arr[0],cmap='Greys_r',interpolation='none', vmin=0, vmax=5, extent=(originX,originX+pixelWidth*cols,originY+pixelHeight*rows,originY))
imFOG2 = axFOG2.imshow(fog2_arr[0],cmap='Greys_r',interpolation='none', vmin=0, vmax=5, extent=(originX,originX+pixelWidth*cols,originY+pixelHeight*rows,originY))
#axNBR.colorbar()

print 'Initted figure'

# The animation function: called to produce a frame for each generation.
def animate(i):
	#print 'setting im'
	ai = animate.Iter
	imDNBR.set_data(animate.dnbr_arr[ai])
	imNBR.set_data(animate.nbr_arr[ai])
	#imMNBR.set_data(animate.mnbr_arr[ai])
	imKNBR.set_data(animate.knbr_arr[ai])
	imNDVI.set_data(animate.ndvi_arr[ai])
	imDNDVI.set_data(animate.dndvi_arr[ai])
	imKNDVI.set_data(animate.kfndvi_arr[ai])
	imVIR.set_data(animate.thin_arr[ai])
	imFOG.set_data(animate.fog_arr[ai])
	imFOG2.set_data(animate.fog2_arr[ai])
	#imCLD.set_data(animate.cloud_arr[ai])
	if len(animate.hs_show_arr[ai]) > 0:
		#scatterDNBR.set_offsets(np.array([[x[0] for x in animate.hs_show_arr[ai]],[x[1] for x in animate.hs_show_arr[ai]]]))
		scatterDNBR.set_offsets(np.array([[x[0],x[1]] for x in animate.hs_show_arr[ai]]))
		scatterDNBR.set_array(np.array([x[2] for x in animate.hs_show_arr[ai]]))
	else:
		scatterDNBR.set_offsets(np.zeros((2,1)))
		scatterDNBR.set_array(np.zeros(1))
	#ttl.set_text('Iteration: %d (1 simulated second per iteration)'%(animate.Iter))
	if animate.Iter < i-1:
		animate.Iter += 1
#animate.raster = rast_arr
#animate.arr = arr_arr
startoffset = 0 # 650
animate.dndvi_arr = dndvi_arr[startoffset:]
animate.ndvi_arr = ndvi_arr[startoffset:]
animate.kfndvi_arr = kfndvi_arr[startoffset:]
animate.knbr_arr = kfnbr_arr[startoffset:]
#animate.mnbr_arr = farr_arr[startoffset:]
animate.nbr_arr = nbr_arr[startoffset:]
animate.dnbr_arr = dnbr_arr[startoffset:]
animate.thin_arr = thin_arr[startoffset:] #virnir_arr[startoffset:]
animate.fog_arr = gross_arr[startoffset:] #fog_arr[startoffset:]
animate.fog2_arr = fog2_arr[startoffset:]
animate.cloud_arr = KGain_arr[startoffset:] #thin_arr #singlerefl_arr[startoffset:]
animate.hs_show_arr = hs_show_arr[startoffset:]
animate.Iter = 0

print 'Set animate'
print len(animate.dndvi_arr)
print i

interval = 200
# Bind our grid to the identifier X in the animate function's namespace.
#Writer = animation.writers['ffmpeg']
#writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
anim = animation.FuncAnimation(fig, animate, interval=interval, repeat=False, frames=len(dndvi_arr))
#anim.save('dnbr_filtered_w_hs_synth.mp4',writer=writer)
plt.show()
