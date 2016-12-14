import pyximport
import numpy as np
pyximport.install(setup_args={'include_dirs':[np.get_include()]})
from cylaplace_9pts import cy_laplacian
import numexpr as ne
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import animation
from matplotlib import colors
from matplotlib.dates import datestr2num
import gdal
import sys
from filterpy.kalman import EnsembleKalmanFilter as EnKF
import bisect

# Constants from Mendel paper
_k = 2.1360e-1 # m^2 s^-1 K^-3
_A = 1.8793e2 # K s^-1 # was e2
_B = 5.5849e2 # K
#_C = 4.8372e-5 # K^-1
_C = 4.8372e-5 # K^-1
_Cs = 0.1625 # s^-1
_lambda = 0.027
_beta = 0.4829
_Ta = 25 + 273.15 #373.15
_Tal = _Ta - 0.01
_dx = 5.0 # metres
#_dy = 5 # metres
_dt = 15 # seconds 
#_kd = _dt *_k/(_dx*_dy)
LIGHTNING = 175 # + 273.15 # Kelvin
# The initial fraction of the forest occupied by trees.
forest_fraction = 0.7 
# Probability of new tree growth per empty cell, and of lightning strike.
p, f = 0.05, 0.000001
# Forest size (number of cells in x and y directions).
nx, ny = 1200, 1200
# Interval between frames (ms).
interval = 1

gausskernel43x43 = np.loadtxt('gk43.txt')

class State:
	def __init__(self, **kwds):
		self.__dict__.update(kwds)


# There should also be a hx() for DNBR or something to do with burn ratio
# This base class may not be necessary because
# Python is duck-typed
class Measurement:
	def __init__(self):
		"""
		Default constructor for Measurment
		"""
		self.default = True
	def hx(state):
		"""
		Return computed value for this measurement
		i.e. Hx in the EnKF literature
		This is the C in O-C intuition
		Subclasses override this virtual function
		"""
		return 0

class Point2D:
	def __init__(self,x,y):
		self.x = x
		self.y = y

class Rect2D:
	"""
	Agnostic to local or global.
	"""
	def __init__(self,minx,maxx,miny,maxy):
		"""
		Define x as Easting
		       y as Northing
		"""
		self.minx = minx
		self.maxx = maxx
		self.miny = miny
		self.maxy = maxy
	def inside(self,pt):
		return (self.minx < pt.x < self.maxx and self.miny < pt.y < self.maxy)
	def area(self):
		return (self.maxx - self.minx) * (self.maxy - self.miny)
	def localPoint(self,pt,scale=1):
		return Point2D((pt.x - self.minx)*scale, (pt.y - self.miny)*scale)
	def shape(self):
		return (self.maxx - self.minx, self.maxy - self.miny)
	@classmethod
	def fromCircle(cls,centre,radius):
		return cls(centre.x - radius, centre.x + radius, centre.y - radius, centre.y + radius)
	
class LocalTransformation2D:
	"""
	Define a mapping from one coordinate system to local grid coordinates based on a rect.
	e.g. Albers coordinates to local grid coordinates for the analysis area
	"""
	def __init__(self,sourceRect,scale=1):
		"""
		Scale is a float
		translation is a Point2D
		"""
		self.sourceRect = sourceRect
		self.scale = scale
	def TransformPointToLocal(self,src_pt):
		return Point2D((src_pt.x - self.sourceRect.minx) * scale,(src_pt.y - self.sourceRect.miny) * scale)
	def getTargetRect(self):
		(maxx,maxy) = sourceRect.shape()
		maxx *= scale
		maxy *= scale
		return Rect2D(0,maxx,0,maxy)

class RectTransformation2D:
	"""
	Define a mapping from one coordinate system to another based on two rects.
	e.g. Albers coordinates to local grid coordinates for the analysis area
	source rect and target rect bounds are defined in same coord system
	"""
	def __init__(self,sourceRect,targetRect,scale=1):
		"""
		Scale is a float
		"""
		self.sourceRect = sourceRect
		self.targetRect = targetRect
		self.scale = scale
	def getTranslation(self):
		self.tr = Point2D(self.targetRect.minx - self.sourceRect.minx,self.targetRect.miny - self.sourceRect.miny)
		return self.tr
	def TransformLocalPointToLocalPoint(self,src_pt):
		"""
		Transform local coordinates in sourceRect into local coordinates for targetRect
		"""
		tr = self.getTranslation()
		return Point2D((src_pt.x - tr.x) * scale,(src_pt.y - tr.y) * scale)
	def getTargetLocalRect(self):
		"""
		get target rect in local coordinates to the target rect, i.e. top left is 0,0
		"""
		(maxx,maxy) = targetRect.shape()
		maxx *= scale
		maxy *= scale
		return Rect2D(0,maxx,0,maxy)


def GetExtent(gt,cols,rows):
	''' Return list of corner coordinates from a geotransform

		@type gt:   C{tuple/list}
		@param gt: geotransform
		@type cols:   C{int}
		@param cols: number of columns in the dataset
		@type rows:   C{int}
		@param rows: number of rows in the dataset
		@rtype:	C{[float,...,float]}
		@return:   coordinates of each corner
	'''
	return Rect2D(gt[0],gt[0]+(cols*gt[1])+(rows*gt[2]), gt[3], gt[3]+(cols*gt[4])+(rows*gt[5]))

def GetPixelSize(gt):
	return Point2D(gt[1],gt[5])

def hx(T,S,V,obstype):
	"""
	Observation function creates synthetic data from the state
	Obstype is the type of observation
	e.g. mean temperature within a 2km x 2km H8 pixel
	"""
	# run h(x) = HX + f

def hx_PixelTemp(T,bounds):
	"""
	mean temperature within bounds extents
	bounds is a set (minx, maxx, miny, maxy) defined over the domain of T 
		i.e. local grid coords for T, not the projection coords. 
		Requires conversion from proj coords say Albers EPSG:3577 to local T grid coords.
	"""
	return T[bounds[0]:bounds[1],bounds[2]:bounds[3]].sum() / ((bounds[1]-bounds[0]) * (bounds[3] - bounds[2]))

def hx_HotspotTemp(T,bounds):
	"""
	max temperature within bounds extents
	bounds is a set (minx, maxx, miny, maxy) defined over the domain of T 
		i.e. local grid coords for T, not the projection coords. 
		Requires conversion from proj coords say Albers EPSG:3577 to local T grid coords.
	"""
	return T[bounds.minx:bounds.maxx,bounds.miny:bounds.maxy].max()

def hx_FuelLoad(S,bounds):
	"""
	TBD
	"""
	return S[bounds[0]:bounds[1],bounds[2]:bounds[3]].sum() / ((bounds[1]-bounds[0]) * (bounds[3] - bounds[2]))


class HotspotMeasurement(Measurement):
	def __init__(self,centre,radius,temp,epoch):
		self.temp = temp
		self.centre = centre
		self.radius = radius
		self.epoch = epoch
		# TODO add epoch?
	def hx(state):
		"""
		return computed hotspot (i.e. max temp) from within bounds or mask
		"""
		localBounds = Rect2D.fromCircle(state.Extent.localPoint(self.centre,scale=(1.0/state.dx)),self.radius)
		# get max temp from state.T within localBounds
		return hs_HotspotTemp(state.T,localBounds)
		
class PixelMeasurement(Measurement):
	def __init__(self,centre,radius,temp,epoch):
		self.temp = temp
		self.centre = centre
		self.radius = radius
		self.epoch = epoch
	def hx(state):
		"""
		return computed average temp (i.e. max temp) from within bounds or mask
		"""
		localBounds = Rect2D.fromCircle(state.Extent.localPoint(self.centre,scale=(1.0/state.dx)),self.radius)
		# get max temp from state.T within localBounds
		return hs_PixelTemp(state.T,localBounds)
		

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

dr='/g/data/r78/lsd547/H8/WA/2016/01/06/'
hs_h8_file='WA_jan2016_H8_ALBERS.csv'
hs_MODIS_file='WA_jan2016_MODIS_VIIRS_ALBERS.csv'

#test_extents_E=(-1487906,-1480756)
#test_extents_N=(-3676644,-3683126)

# min/max E, min/max N
test_extents = Rect2D(-1487906,-1480756,-3676644,-3683126)

# define how the extents map to a local grid for analysis
# Source grid is in Albers in metres (ignore distortions) and destination is
# a 2D grid with 5x5 metre cell size
gridtrans = LocalTransformation2D(test_extents,(1/_dt))

H8_B07 = []

#ymd = '20160106'
y = '2016'
m = '01'
d = '06'

# Load band 7 imagery
for hour in range(24):
	for tenminute in range(6):
		minute = tenminute * 10
		
		#fn = '20160106_1910_B07_Aus.tif'
		fn = '%s%s%s_%02d%02d_B07_Aus.tif'%(y,m,d,hour,minute)
		#print 'opening ' + dr + fn
		try:
			r = gdal.Open(dr+fn)
			# TODO crop to test extents
			a = np.array(r.GetRasterBand(1).ReadAsArray())
			tr = r.GetGeoTransform()
			#print 'loaded'
			H8_B07.append( (utc2num('%s-%s-%sT%02d:%02d:00Z'%(y,m,d,hour,minute)),tr,r,a) )
			#i += 1
		except AttributeError, e:
			print e
			print "Unexpected error:", sys.exc_info()[0]
			print 'File for time %02d%02d does not exist'%(hour,minute)

#Load hotspots
hs_h8 = np.loadtxt(hs_h8_file,skiprows=1,delimiter=',',usecols=(0,1,2,3,4,5,6,7,8),converters={2:datestr2num})
hs_MODIS = np.loadtxt(hs_MODIS_file,skiprows=1,delimiter=',',usecols=(0,1,19,20,13,15,26,27,28,29,30,31,32), converters={15:sv2num,19:utc2num,20:utc2num,28:floatOrBlank,29:utc2num,30:floatOrBlank,31:floatOrBlank}) # X, Y, id, satellite_, start_dt, end_dt, lat, lon, temp_K, datetime, power, confidence, age_hours 
#hs_MODIS = np.loadtxt(hs_MODIS_file,skiprows=1,delimiter=',',usecols=(0,1,13,15,19,20,29,26,27,28), converters={15:sv2num,19:utc2num,20:utc2num,29:utc2num,28:temp2num}) # X, Y, id, satellite_, start_dt, end_dt, lat, lon, temp_K, datetime, power, confidence, age_hours 
#hs_MODIS = np.loadtxt(hs_MODIS_file,skiprows=1,delimiter=',',usecols=(0,1,13,15), converters={15:sv2num,19:utc2num,20:utc2num,29:utc2num}) # X, Y, id, satellite_, start_dt, end_dt, lat, lon, temp_K, datetime, power, confidence, age_hours 

def contains(minx,maxx,miny,maxy,x,y):
	return (minx <= x <= maxx and miny <= y <= maxy)

#Delete MODIS/VIIRS/AVHRR entries that are outside the analysis area
index = 0
idx = []
for row in hs_MODIS:
	pt = Point2D(row[0],row[1])
	#if not contains(test_extents_E[0], test_extents_E[1], test_extents_N[0], test_extents_N[1], row[0], row[1]):
	if not test_extents.inside(pt):
		# FIXME this will fail because i will be too big! Find another way to delete rows.
		idx.append(index)
	index += 1

hs_MODIS = np.delete(hs_MODIS, idx, axis=0)

#Delete H8 entries that are outside the analysis area
index = 0
idx = []
for row in hs_h8:
	pt = Point2D(row[0],row[1])
	#if not contains(test_extents_E[0], test_extents_E[1], test_extents_N[0], test_extents_N[1], row[0], row[1]):
	if not test_extents.inside(pt):
		# FIXME this will fail because i will be too big! Find another way to delete rows.
		idx.append(index)
	index += 1

hs_h8 = np.delete(hs_h8, idx, axis=0)

# sort arrays
hs_MODIS[hs_MODIS[:,2].argsort()] #3rd column is start_dt
hs_h8[hs_h8[:,2].argsort()] #3rd column is utc time
H8_B07 = sorted(H8_B07,key=lambda x: x[0])

# The reason why OrderedDict is used but data is kept in Tuples is because multiple hotspots exist for one epoch, and key,value pairs where key=epoch would either require a list for values or would only allow one value.
#hs_MODIS_meas = []
hs_MODIS_radius = 0.5 * 500.0 / _dx

#hs_h8_meas = []
hs_h8_radius = 0.5 * 2000.0 / _dx

#H8_B07_meas = []
H8_B07_radius = 500.0 / _dx

all_meas_dict = dict()
all_meas = [] # list of tuples (epoch,[meas,meas,...])
sorted_epochs = []

# Need to use a dict for insertion first because the key check is fast and easy
def addMeas(epoch,meas):
	if epoch not in all_meas_dict.keys():
		all_meas_dict[epoch] = []
	all_meas_dict[epoch].append(meas)
	
# Convert dict to list for sequential retrieval of measurements by epoch
def sortMeas():
	sorted_epochs = sorted(all_meas_dict.keys()) # global
	#all_meas = OrderedDict(sorted(all_meas, key=lambda m: m[0]))
	for e in sorted_epochs:
		all_meas.append((epoch,all_meas_dict[epoch]))

def getNextEpoch(currentEpoch):
	index = bisect.bisect_right(sorted_epochs, currentEpoch) # TODO set lo and hi to moving index
	if index < len(sorted_epochs):
		# there is more data
		return sorted_epochs[index]
	else:
		raise Exception("Processing has finished") # Bad bad bad don't do this in production

def getMeasForEpoch(epoch):
	"""
	Return array of measurements
	"""
	return all_meas_dict[epoch]
	
	

# create measurement objects
for row in hs_MODIS:
	# fixme make radius in grid cell counts
	epoch = row[2]
	#hs_MODIS_meas.append((row[2],HotspotMeasurement(Point2D(row[0],row[1]),hs_MODIS_radius,row[8],row[2])))
	addMeas(epoch,HotspotMeasurement(Point2D(row[0],row[1]),hs_MODIS_radius,row[8],row[2]))
#hs_MODIS_meas = OrderedDict(sorted(hs_MODIS_meas, key=lambda m: m[0]))

for row in hs_h8:
	# fixme make radius in grid cell counts
	#hs_h8_meas.append((row[2],HotspotMeasurement(Point2D(row[0],row[1]),hs_h8_radius,row[7],row[2]))) # TODO also use FRP and Fire size
	addMeas(epoch,HotspotMeasurement(Point2D(row[0],row[1]),hs_h8_radius,row[7],row[2])) # TODO also use FRP and Fire size
#hs_h8_meas = OrderedDict(sorted(hs_h8_meas, key=lambda m: m[0]))

for e,tr,r,a in H8_B07:
	cols = r.RasterXSize
	rows = r.RasterYSize
	rasterbounds = GetExtent(tr,cols,rows)
	pixSize = GetPixelSize(tr)
	# we're going from the pixSize of the raster (500m by 500m) to the grid cell size in metres (5m by 5m)
	trans = RectTransformation2D(rasterbounds,test_extent,scale=(pixSize.x/_dx))
	# loop through each pixel and if it is in the test_extent bounds add it to the measurement list
	for col in xrange(cols):
		for row in xrange(rows):
			# Transform from raster coordinate system to grid coordinate system
			gridPixPt = trans.TransformLocalPointToLocalPoint(Point2D(col,row))
			if trans.getTargetLocalRect().inside(gridPixPt):
				# FIXME confirm that col,row referencing order is correct visually
				#H8_B07_meas.append((e,PixelMeasurement(gridPixPt,trans.scale*0.5,a[col][row],e)))
				addMeas(e,PixelMeasurement(gridPixPt,trans.scale*0.5,a[col][row],e))
	 
#H8_B07_meas = OrderedDict(sorted(H8_B07_meas, key=lambda m: m[0]))

sortMeas()

def iterate(T,S,V,dt):
	"""Iterate the forest according to the forest-fire rules. No wind term."""
	(ny, nx) = S.shape
	kd = dt *_k

	laplace_T = np.zeros((ny,nx))
	#ndimage.filters.laplace(T,output=laplace_T,mode='constant',cval=_Ta)
	#laplace_T /= _dx**2
	cy_laplacian(T,laplace_T,_dx**2,_Ta)
	#cy_laplacian(T,laplace_T,_dx,_Ta)

	[grad_x_T,grad_y_T] = np.gradient(T)
	[vx,vy] = V * dt

	SexpB_T = ne.evaluate('S * exp(-_B / (T - _Tal))')
	T_dT_dt = ne.evaluate('T + (kd * laplace_T + vx * grad_x_T + vy * grad_y_T + dt * _A * SexpB_T - dt * _A * _C * (T - _Tal))')
	S_dS_dt = ne.evaluate('S - dt * _Cs * SexpB_T')

	return (T_dT_dt,S_dS_dt)

def fx(T,S,V,dt):
	# split x into T,S,V
	# propagate by dt
	while dt > _dt:
		(T,S) = iterate(T,S,V,_dt)
		dt -= _dt
	(T,S) = iterate(T,S,V,dt)
	return (T,S,V)

# TODO Write EnKF core here.

	

# Note to Friday Laurence: due to "morphing"? data availability, somehow use predictions as data but don't affect covariance somehow. 
# not sure if this is done in the HA() update step? Observed minus calculated, but observed is equal to calculated.
# Update: morphing re Mendel's paper refers to image registration

# for presentation:
#    movie with the following panes:
#     1) Vector fire front with CI over Landsat burn scar
#     2) Mean temperature
#     3) Variance of temperature
#     4) Mean fuel load
#     5) Variance of fuel load
#     6) Mean wind speed and direction
#     7) Variance of wind speed and direction
#     8) Observed Band 7 or 14 temperature
#     9) Residual Band 7 or 14 temperature
#     10) Observed hotspot temperature
#     11) Residual hotspot temperature

#EnKF procedure
# From https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/Appendix-E-Ensemble-Kalman-Filters.ipynb
# 1) Init state with starting conditions, N = number of ensemble members (i.e. sigmas)
# 2) Create an ensemble of initted state. Do this as a for-each and MPI branch from here.
# 3) Let x = the mean state across all ensemble members (i.e. sigma using rlabbe's pyfilter terminology)
# 4) let P = covariance = [sigma - x][sigma - x]^transposed, so P is NxN rank N. Use numpy.outer to computer outer product
# 5) For each sigma, for each observation
#        hx_sigma_obs = hx(sigma, observation)
# 6) z_mean = average hx_sigma_obs over all sigmas for each observation (e.g. per-pixel average over ensemble)
# 7) Update Pzz and Pxz... then both together produce Kalman Gain = Pxz * inv(Pzz)
# 8) For each sigma, update via sigma += KalmanGain * [z_mean - hx + Vr] wher Vr is the perturbation added to the sigma
# 9) update mean by summing all sigmas together (T,S,V) and dividing by N. Pointwise, cheap and easy.
# 10) Update covariance P by P -= KalmanGain*Pzz*KalmanGain^transpose




# 1) Init state with starting conditions, N = number of ensemble members (i.e. sigmas)
N = 64
T_stddev = 10 # degrees
_epoch = getNextEpoch(0) # get first epoch
# get first lot of measurements for this epoch and check for hotspots!
# For now don't use MPI, use a list of starting states

#sigmas = [State(T=np.ones(ny,nx)*_Ta,S=np.ones((ny,nx)),V=np.array([0.0,0.0])) for i in range(N)]

def randomBump(T,temp):
	# Simulate a lightning strike for initial conditions
	(ny,nx) = T.shape
	ks = 21 # kernel size = 2 * ks + 1
	#if np.random.random() <= f * (nx - 2*ks) * (ny  - 2*ks):
	ix = np.random.randint(ks,nx-ks)
	iy = np.random.randint(ks,ny-ks)
	T[iy-ks:iy+ks+1,ix-ks:ix+ks+1] += temp * gausskernel43x43 / np.amax(gausskernel43x43)
	

def generateState(Ta_stddev):
	# Initialize the forest grid.
	S  = np.zeros((ny, nx))
	S[1:ny-1, 1:nx-1] = np.random.randint(0, 2, size=(ny-2, nx-2))
	S[1:ny-1, 1:nx-1] = np.random.random(size=(ny-2, nx-2)) < forest_fraction
	# Initialize the ambient temperature grid.
	T  = np.ones((ny, nx)) * _Ta + np.random.normal(0,Ta_stddev,(ny,nx))
	V = np.array([0.0,0.0])
	return State(T=T,S=S,V=V,Extent=test_extents)
	
# 2) Create an ensemble of initted state. Do this as a for-each and MPI branch from here.
#T_sigmas = multivariate_normal(mean=T.reshape(ny*nx),cov=TP.reshape(ny*nx), size=N)
# create ensemble members i.e. sigmas

# First state will be noisy on purpose. Try to get best fit from the noise given the measurements.
sigmas = [generateState(_Ta_stddev) for i in range(N)]

# 3) Let x = the mean state across all ensemble members (i.e. sigma using rlabbe's pyfilter terminology)
def reduceAdd(li):
	return reduce(lambda x,y: x+y, li)

X_mean_state = State(T=reduceAdd([s.T for s in sigmas])/N, S=reduceAdd([s.S for s in sigmas])/N, V=reduceAdd([s.V for s in sigmas])/N)

# 4) let P = covariance = [sigma - x][sigma - x]^transposed, so P is NxN rank N. Use numpy.outer to computer outer product. OPTIONAL. P is huge.

# 5) For each sigma, for each observation
#        hx_sigma_obs = hx(sigma, observation)

# get measurements for epoch
meas_at_epoch = getMeasForEpoch(_epoch)



for s in sigmas:
	
	
#	for meas in hs_h8:
#		# convert line to useful observation.
#	for line in hs_MODIS:
#		# convert line to useful observation.
#	for (epoch,arr) in H8_B07_arr:


#maxEpoch = 999999999999
#
#def getNextEpoch(currentEpoch):
#	"""
#	Usage:
#		nextEpoch = getNextEpoch(epoch)
#		if nextEpoch != maxEpoch:
#			dt = nextEpoch - epoch
#		else:
#			exit
#	"""
#	hs_h8_index = np.searchsorted(hs_h8_meas[:,], currentEpoch, side='right')
#	hs_MODIS_index = np.searchsorted(hs_MODIS_meas[:,2], currentEpoch, side='right')
#	H8_B07_index = bisect.bisect_right(H8_B07_meas[:,0], currentEpoch) # TODO set lo and hi to moving index
#	
#	nextEpoch = maxEpoch
#	
#	(rows,cols) = hs_h8.shape
#	if hs_h8_index < rows:
#		# There exist measurements after currentEpoch for hs_h8
#		nextEpoch = min(nextEpoch, hs_h8[hs_h8_index,2])
#
#	(rows,cols) = hs_MODIS.shape
#	if hs_MODIS_index < rows:
#		# There exist measurements after currentEpoch for hs_MODIS
#		nextEpoch = min(nextEpoch, hs_MODIS[hs_MODIS_index,2])
#
#	if H8_B07_index < len(H8_B07[:,0]:
#		# There exist measurements after currentEpoch for H8_B07
#		nextEpoch = min(nextEpoch, H8_B07[H8_B07_index,0])
#
#	return nextEpoch
			
		
def getHSH8Measurements(s,epoch):
	"""
	Args: s = State object, currentEpoch is number representing UTC time
	Returns a [list of measurements] that occur at epoch
	"""
	measList = []
	# get indices of next epoch measurements 
	hs_h8_left_index = np.searchsorted(hs_h8[:,2], [currentEpoch], side='left')
	hs_h8_right_index = np.searchsorted(hs_h8[:,2], [currentEpoch], side='right')
	
	if hs_h8_left_index < hs_h8_right_index:
		# There exist measurements at epoch for hs_h8
		i = hs_h8_left_index
		while i < hs_h8_right_index:
			measList.append(hs_h8[i,:])
			i += 1
	
	return measlist

def getHSMODISMeasurements(s,currentEpoch):
	"""
	Args: s = State object, currentEpoch is number representing UTC time
	Returns a [list of measurements] that occur at epoch
	"""
	# get indices of next epoch measurements 
	hs_MODIS_index = np.searchsorted(hs_MODIS[:,2], [currentEpoch], side='right')
	
	(rows,cols) = hs_MODIS.shape
	if hs_MODIS_index < rows:
		# There exist measurements after currentEpoch for hs_MODIS
	

# 6) z_mean = average hx_sigma_obs over all sigmas for each observation (e.g. per-pixel average over ensemble)
# 7) Update Pzz and Pxz... then both together produce Kalman Gain = Pxz * inv(Pzz)
# 8) For each sigma, update via sigma += KalmanGain * [z_mean - hx + Vr] wher Vr is the perturbation added to the sigma
# 9) update mean by summing all sigmas together (T,S,V) and dividing by N. Pointwise, cheap and easy.
# 10) Update covariance P by P -= KalmanGain*Pzz*KalmanGain^transpose

	

# Spatial perturbations: create a "Kernel" of 7x7 spatial perturbations with the centre being 0,0

def makeSimPlotWindow(T,S,V):
	# FIXME remove animation and write images to disk if desired
	fig = plt.figure(figsize=(50/3, 12.5))
	gs = gridspec.GridSpec(1,2, width_ratios=[1,1])
	axS = fig.add_subplot(gs[0,0])
	axS.set_axis_off()
	#imS = axS.imshow(S, cmap=cmap, norm=norm)#, interpolation='nearest')
	imS = axS.imshow(S, cmap='gray', vmin=0, vmax=1)#, interpolation='nearest')
	axT = fig.add_subplot(gs[0,1])
	axT.set_axis_off()
	#imT = axT.imshow(T, cmap=cmap, norm=norm)#, interpolation='nearest')
	imT = axT.imshow(T, cmap='jet', vmin=_Tal, vmax=1000, interpolation='none')#, interpolation='nearest')
	ttl = axS.text(.5, 1.05, '', transform = axS.transAxes, va='center')
	
	# The animation function: called to produce a frame for each generation.
	def animate(i):
		imT.set_data(animate.T)
		imS.set_data(animate.S)
		animate.V += np.random.rand(2,) * 0.01 - 0.005
		ttl.set_text('Iteration: %d (%d simulated second per iteration), Wind velocity = (%f, %f)'%(animate.Iter,_dt,animate.V[0],animate.V[1]))
		animate.Iter += 1
		(animate.T,animate.S) = iterate(animate.T,animate.S,animate.V)
	
	# Bind our grid to the identifier X in the animate function's namespace.
	animate.S = S
	animate.T = T
	animate.V = V
	animate.Iter = 0
	plt.tight_layout(pad=0.01, w_pad=0.01, h_pad=0.01)
	anim = animation.FuncAnimation(fig, animate, interval=interval)
	plt.show()

