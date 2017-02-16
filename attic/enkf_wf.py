import pyximport
#from functools import partial
#import multiprocessing as mp
#from multiprocessing import Pool, cpu_count
#from multiprocessing.pool import ApplyResult
#from copy_reg import pickle
#from types import MethodType



import math
import time
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
#from filterpy.kalman import EnsembleKalmanFilter as EnKF
import bisect

IS_PY2 = sys.version_info < (3, 0)

if IS_PY2:
    from Queue import Queue
else:
    from queue import Queue

from threading import Thread


Vsigma = 0.0001
# Constants from Mendel paper
_k = 2.1360e-1 # m^2 s^-1 K^-3
_A = 1.8793e2 # K s^-1 # was e2
_B = 5.5849e2 # K
#_C = 4.8372e-5 # K^-1
_C = 4.8372e-5 # K^-1
_Cs = 0.1625 # s^-1
_lambda = 0.027
_beta = 0.4829
_Ta = 50 + 273.15 # initialization mean temperature
_Tal = 25 + 273.15 # ambient temperature
_Talclip = _Tal + 1 # clip measurements
_dx = 5.0 # metres
#_dy = 5 # metres
_dt = 15 # seconds 
#_kd = _dt *_k/(_dx*_dy)
LIGHTNING = 175 # + 273.15 # Kelvin
# The initial fraction of the forest occupied by trees.
forest_fraction = 0.99 
# Probability of new tree growth per empty cell, and of lightning strike.
p, f = 0.05, 0.000001
# Interval between frames (ms).
interval = 1
#test_extents_E=(-1487906,-1480756)
#test_extents_N=(-3676644,-3683126)

N = 16 # Number of members in the ensemble
#_T_stddev = 20 # degrees (not used afaik)
_Ta_stddev = 80
Tnoise = 5.0 # per ensemble member noise added at start of each simulation run

def _pickle_method(method):
	func_name = method.im_func.__name__
	obj = method.im_self
	cls = method.im_class
	return _unpickle_method, (func_name, obj, cls)

def _unpickle_method(func_name, obj, cls):
	for cls in cls.mro():
		try:
			func = cls.__dict__[func_name]
		except KeyError:
			pass
		else:
			break
	return func.__get__(obj, cls)


class Worker(Thread):
	""" Thread executing tasks from a given tasks queue """
	def __init__(self, tasks):
		Thread.__init__(self)
		self.tasks = tasks
		self.daemon = True
		self.start()

	def run(self):
		while True:
			func, args, kargs = self.tasks.get()
			try:
				func(*args, **kargs)
			except Exception as e:
				# An exception happened in this thread
				print(e)
			finally:
				# Mark this task as done, whether an exception happened or not
				self.tasks.task_done()

class ThreadPool:
	""" Pool of threads consuming tasks from a queue """
	def __init__(self, num_threads):
		self.tasks = Queue(num_threads)
		for _ in range(num_threads):
			Worker(self.tasks)

	def add_task(self, func, *args, **kargs):
		""" Add a task to the queue """
		self.tasks.put((func, args, kargs))

	def map(self, func, args_list):
		""" Add a list of tasks to the queue """
		for args in args_list:
			self.add_task(func, args)
	def wait_completion(self):
		""" Wait for completion of all the tasks in the queue """
		self.tasks.join()


class State(object):
	def __init__(self, **kwds):
		self.__dict__.update(kwds)
	def LocalExtent(self):
		(w,h) = self.Extent.shape()
		w /= self.dx
		h /= self.dx
		return Rect2D(0,w,0,h)
	def iterate(self,dt):
		"""Iterate the forest according to the forest-fire rules. No wind term."""
		(T,S,V) = (self.T,self.S,self.V) # expand
		(ny, nx) = S.shape
		kd = dt *_k
	
		laplace_T = np.zeros((ny,nx))
		#ndimage.filters.laplace(T,output=laplace_T,mode='constant',cval=_Ta)
		#laplace_T /= _dx**2
		cy_laplacian(T,laplace_T,_dx**2,_Tal)
		#cy_laplacian(T,laplace_T,_dx,_Ta)
	
		[grad_x_T,grad_y_T] = np.gradient(T)
		[vx,vy] = V * dt
	
		SexpB_T = ne.evaluate('S * exp(-_B / (T - _Tal))')
		self.T = ne.evaluate('T + (kd * laplace_T + vx * grad_x_T + vy * grad_y_T + dt * _A * SexpB_T - dt * _A * _C * (T - _Tal))')
		self.S = ne.evaluate('S - dt * _Cs * SexpB_T')
		del(SexpB_T)
		del(grad_x_T)
		del(grad_y_T)
		del(laplace_T)
	def setFxParams(self,dt,step_dt,Tnoise):
		self.dt = dt
		self.step_dt = step_dt
		self.Tnoise = Tnoise
	def fx(self):
		# split x into T,S,V
		# add noise to the starting model to increase the ensemble variance
		(ny,nx) = self.T.shape
		self.T += ndimage.filters.gaussian_filter(np.random.normal(0,self.Tnoise,(ny,nx)),2,mode='nearest')
		np.clip(self.V,-2*Vsigma, 2*Vsigma,out=self.V)
		np.clip(self.T,_Talclip,1000,out=self.T)
		# propagate by dt
		while self.dt > self.step_dt:
			#print 'Simulating ' + str(_dt) + ' seconds'
			self.iterate(self.step_dt)
			self.dt -= self.step_dt
			np.clip(self.T,_Talclip,1000,out=self.T)
		#print 'Simulating ' + str(self.dt) + ' seconds'
		self.iterate(self.dt)
	def perturbTemp(self,b,p):
		print "Perturbing temperature." # Bounds shape: " + str(b) + " Kernel shape: " + str(p)
		self.T[b.minyi():b.maxyi(),b.minxi():b.maxxi()] += p
	def perturbFuel(self,b,p):
		print "Perturbing fuel." # Bounds shape: " + str(b) + " Kernel shape: " + str(p)
		self.S[b.minyi():b.maxyi(),b.minxi():b.maxxi()] += p

def doFx(s):
	s.fx()
	#return s

# pickle line required for multiprocessing
#pickle(MethodType, _pickle_method, _unpickle_method)

# There should also be a hx() for DNBR or something to do with burn ratio
# This base class may not be necessary because
# Python is duck-typed
class Measurement:
	def __init__(self):
		"""
		Default constructor for Measurment
		"""
		self.default = True
#	def hx(state):
#		"""
#		Return computed value for this measurement
#		i.e. Hx in the EnKF literature
#		This is the C in O-C intuition
#		Subclasses override this virtual function
#		"""
#		return 0

class Point2D:
	def __init__(self,x,y):
		self.x = x
		self.y = y
	def fromOrigin(self,o):
		return Point2D(self.x - o.x, self.y - o.y)
	def __str__(self):
		return str((self.x,self.y))

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
	def minxi(self):
		return int(self.minx)
	def maxxi(self):
		return int(self.maxx + 0.5)
	def minyi(self):
		return int(self.miny)
	def maxyi(self):
		return int(self.maxy + 0.5)
	def inside(self,pt):
		return (self.minx < pt.x < self.maxx and self.miny < pt.y < self.maxy)
	def area(self):
		#print "Min x" + str(self.minx)
		#print "Max x" + str(self.maxx)
		#print "Length x " + str(self.maxx - self.minx)
		#print "Length y " + str(self.maxy - self.miny)
		return ((self.maxx - self.minx) * (self.maxy - self.miny))
	def localPoint(self,pt,scale=1):
		return Point2D((pt.x - self.minx)*scale, (pt.y - self.miny)*scale)
	def shape(self):
		return (self.maxx - self.minx, self.maxy - self.miny)
	def topLeft(self):
		return Point2D(self.minx, self.miny)
	def clipTo(self,r):
		return Rect2D(max(self.minx,r.minx),min(self.maxx,r.maxx),max(self.miny,r.miny),min(self.maxy,r.maxy))
	def clipToInt(self,r):
		return Rect2D(max(self.minxi(),r.minxi()),min(self.maxxi(),r.maxxi()),max(self.minyi(),r.minyi()),min(self.maxyi(),r.maxyi()))
	def __str__(self):
		return "Min (" + str(self.minx) + ", " + str(self.maxx) + "), Max (" + str(self.miny) + ", " + str(self.maxy) + ")"
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
	def getGlobalSourceToLocalTargetTranslation(self):
		"""
		global coords wrt source rect
		translate to local coords wrt target rect
		"""
		self.tr = Point2D(-(self.targetRect.minx - self.sourceRect.minx),-(self.targetRect.miny - self.sourceRect.miny))
		return self.tr
	def TransformLocalPointToLocalPoint(self,src_pt):
		"""
		Transform local coordinates in sourceRect into local coordinates for targetRect
		"""
		tr = self.getGlobalSourceToLocalTargetTranslation()
		return Point2D((src_pt.x + tr.x) * self.scale,(src_pt.y + tr.y) * self.scale)
	def getTargetLocalRect(self):
		"""
		get target rect in local coordinates to the target rect, i.e. top left is 0,0
		"""
		(maxx,maxy) = self.targetRect.shape()
		maxx *= self.scale
		maxy *= self.scale
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

def hx_PixelNBR(S,bounds):
	"""
	mean fuel within bounds extents
	bounds is a set (minx, maxx, miny, maxy) defined over the domain of S
		i.e. local grid coords for S, not the projection coords. 
		Requires conversion from proj coords say Albers EPSG:3577 to local S grid coords.
	"""
	return S[bounds.minyi():bounds.maxyi(),bounds.minxi():bounds.maxxi()].sum() / bounds.area()

def hx_PixelTemp(T,bounds):
	"""
	mean temperature within bounds extents
	bounds is a set (minx, maxx, miny, maxy) defined over the domain of T 
		i.e. local grid coords for T, not the projection coords. 
		Requires conversion from proj coords say Albers EPSG:3577 to local T grid coords.
	"""
	#print "HX Bounds " + str(bounds)
	#print "HX Area " + str(bounds.area())
	return T[bounds.minyi():bounds.maxyi(),bounds.minxi():bounds.maxxi()].sum() / bounds.area()

def hx_HotspotTemp(T,bounds):
	"""
	max temperature within bounds extents
	bounds is a set (minx, maxx, miny, maxy) defined over the domain of T 
		i.e. local grid coords for T, not the projection coords. 
		Requires conversion from proj coords say Albers EPSG:3577 to local T grid coords.
	"""
	#print "Size of T: " + str(T.shape)
	print "Bounds to slice: " + str(bounds)
	return T[bounds.minyi():bounds.maxyi(),bounds.minxi():bounds.maxxi()].max()

def hx_FuelLoad(S,bounds):
	"""
	TBD
	"""
	return S[bounds.minyi():bounds.maxyi(),bounds.minxi():bounds.maxxi()].sum() / bounds.area()

def gkern2(rows,cols,rowo,colo, nsig=3):
	"""Returns a 2D Gaussian kernel array."""
	# create nxn zeros
	inp = np.zeros((rows, cols))
	# set element at the middle to one, a dirac delta
	inp[int(rowo), int(colo)] = 1
	# gaussian-smooth the dirac, resulting in a gaussian filter mask
	return ndimage.filters.gaussian_filter(inp, nsig)

class HotspotMeasurement(Measurement):
	def __init__(self,centre,radius,temp,confidence,epoch):
		if temp > 0:
			self.temp = temp
			if confidence > 0:
				self.quality = 0.1 + 0.05 * (100 - confidence) # / 2.69?
			else:
				self.quality = 1
		else:
			self.temp = 500 # ignition?
			self.quality = 2
		self.centre = centre
		self.radius = radius
		self.epoch = epoch
		# TODO add epoch?
	def hx(self,state):
		"""
		return computed hotspot (i.e. max temp) from within bounds or mask
		"""
		localBounds = self.getLocalBounds(state)
		# get max temp from state.T within localBounds
		return hx_HotspotTemp(state.T,localBounds)
	def d(self):
		return max(self.temp,_Talclip)
	def r(self):
		return self.quality
	def getLocalBounds(self,state):
		#print "Local bounds " + str(localBounds)
		# clip to local extents
		localBounds = Rect2D.fromCircle(state.Extent.localPoint(self.centre,scale=abs(1.0/state.dx)),self.radius/state.dx)
		return localBounds.clipToInt(state.LocalExtent())
	def perturbEnsembleState(self,state):
		lb = self.getLocalBounds(state)
		(cols,rows) = lb.shape()
		cols = int(cols)
		rows = int(rows)
		border = 1
		if cols/2 > border and rows/2 > border:
			print "Hotspot perturbation"
			rx = np.random.randint(0+border,cols - border)
			ry = np.random.randint(0+border,rows - border)
			state.perturbTemp(lb,np.random.normal(self.d()-self.hx(state),self.r(),1) * gkern2(rows,cols,ry,rx,nsig=2.5))

class PixelTempMeasurement(Measurement):
	def __init__(self,centre,radius,temp,epoch):
		self.temp = temp
		self.centre = centre
		self.radius = radius
		self.epoch = epoch
		self.quality = 20 # degrees std dev at a guess
	def hx(self,state):
		"""
		return computed average temp from within bounds or mask
		"""
		localBounds = self.getLocalBounds(state)
		# get temp from state.T within localBounds
		return hx_PixelTemp(state.T,localBounds)
	def d(self):
		return max(self.temp,_Talclip)
	def r(self):
		return self.quality
	def getLocalBounds(self,state):
		# clip to local extents
		localBounds = Rect2D.fromCircle(state.Extent.localPoint(self.centre,scale=abs(1.0/state.dx)),abs(self.radius/state.dx))
		return localBounds.clipToInt(state.LocalExtent())
	def perturbEnsembleState(self,state):
		lb = self.getLocalBounds(state)
		(cols,rows) = lb.shape()
		cols = int(cols)
		rows = int(rows)
		state.perturbTemp(lb,ndimage.filters.gaussian_filter(np.random.normal(self.d()-self.hx(state),self.r(),(rows,cols)),2,mode='nearest'))

def sinScale(v):
	return math.sin(np.clip(v,0,1)* math.pi / 2.0)

class PixelNBRMeasurement(Measurement):
	def __init__(self,centre,radius,nbr,epoch):
		self.nbr = nbr
		self.centre = centre
		self.radius = radius
		self.epoch = epoch
		self.quality = 0.1 # 10% std dev at a guess
	def hx(self,state):
		"""
		return computed average fuel load from within bounds or mask
		"""
		localBounds = self.getLocalBounds(state)
		# get nbr from state.S within localBounds
		#return math.sin(hx_PixelNBR(state.S,localBounds) * math.pi / 2.0)
		return sinScale(sinScale(hx_PixelNBR(state.S,localBounds)))
	def d(self):
		return self.nbr
	def r(self):
		return self.quality
	def getLocalBounds(self,state):
		# clip to local extents
		localBounds = Rect2D.fromCircle(state.Extent.localPoint(self.centre,scale=abs(1.0/state.dx)),abs(self.radius/state.dx))
		return localBounds.clipToInt(state.LocalExtent())
	def perturbEnsembleState(self,state):
		lb = self.getLocalBounds(state)
		(cols,rows) = lb.shape()
		cols = int(cols)
		rows = int(rows)
		state.perturbFuel(lb,ndimage.filters.gaussian_filter(np.random.normal(self.d()-self.hx(state),self.r(),(rows,cols)),2,mode='nearest'))
		

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

# Spatial perturbations: create a "Kernel" of 7x7 spatial perturbations with the centre being 0,0

def makeSimPlotWindow(s,s0):
	global fig, gs, imS, imT, imS0, imT0, ttl
	(T,S,V) = (s.T, s.S, s.V)
	fig = plt.figure(figsize=(50/3, 12.5))
	gs = gridspec.GridSpec(2,2, width_ratios=[1,1])
	# FIXME remove animation and write images to disk if desired
	axS = fig.add_subplot(gs[0,0])
	axS.set_axis_off()
	axT = fig.add_subplot(gs[0,1])
	axT.set_axis_off()
	axS0 = fig.add_subplot(gs[1,0])
	axS0.set_axis_off()
	axT0 = fig.add_subplot(gs[1,1])
	axT0.set_axis_off()
	imS = axS.imshow(S, cmap='gray', vmin=0, vmax=1, interpolation='none')#, interpolation='nearest')
	imT = axT.imshow(T, cmap='jet', vmin=_Tal, vmax=1000, interpolation='none')#, interpolation='nearest')
	imS0 = axS0.imshow(s0.S, cmap='gray', vmin=0, vmax=1, interpolation='none')#, interpolation='nearest')
	imT0 = axT0.imshow(s0.T, cmap='jet', vmin=_Tal, vmax=1000, interpolation='none')#, interpolation='nearest')
	ttl = axS.text(.5, 1.05, '', transform = axS.transAxes, va='center')
	plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
	#plt.ion()
	#plt.show()
	#plt.draw()
	#plt.pause(0.05)
	#plt.savefig('start.png')
	#plt.close()

def plotState2(s,s0,e):
	global fig, gs, imS, imT, imS0, imT0, ttl
	#makeSimPlotWindow(s,s0)
	imS.set_data(s.S)
	imT.set_data(s.T)
	imS0.set_data(s0.S)
	imT0.set_data(s0.T)
	ttl.set_text('Iteration: %d, Wind velocity = (%f, %f)'%(s.Iter,s.V[0],s.V[1]))
	#plt.draw()
	plt.pause(0.05)
	#time.sleep(1)
	plt.savefig('out_' + str(e) + '.png')
	plt.close()

def plotState(s,sigmas_,e):
	fig = plt.figure(figsize=(9, 9),dpi=72)
	N_ = len(sigmas_)
	sqrtN_ = int(math.sqrt(N_))
	gs = gridspec.GridSpec(sqrtN_ + 1,2*sqrtN_) #, width_ratios=[1,1])
	# FIXME remove animation and write images to disk if desired
	axS = fig.add_subplot(gs[0,0:sqrtN_])
	axS.set_axis_off()
	axT = fig.add_subplot(gs[0,sqrtN_:(2*sqrtN_)])
	axT.set_axis_off()
	imS = axS.imshow(s.S, cmap='gray', vmin=0, vmax=1, interpolation='none')#, interpolation='nearest')
	imT = axT.imshow(s.T, cmap='jet', vmin=_Tal, vmax=1000, interpolation='none')#, interpolation='nearest')
	ttl = axS.text(.5, 1.05, '', transform = axS.transAxes, va='center')
	ttl.set_text('Iteration: %d, Wind velocity = (%f, %f)'%(s.Iter,s.V[0],s.V[1]))
	axS_ = []
	axT_ = []
	imS_ = []
	imT_ = []
	for i,s_ in enumerate(sigmas_):
		row = i / sqrtN_ + 1
		col = 2*(i % sqrtN_)
		axS_.append(fig.add_subplot(gs[row,col]))
		axS_[i].set_axis_off()
		axT_.append(fig.add_subplot(gs[row,col+1]))
		axT_[i].set_axis_off()
		imS_.append(axS_[i].imshow(s_.S, cmap='gray', vmin=0, vmax=1, interpolation='none'))#, interpolation='nearest')
		imT_.append(axT_[i].imshow(s_.T, cmap='jet', vmin=_Tal, vmax=1000, interpolation='none'))#, interpolation='nearest')
	fig.savefig('out_' + str(e) + '.png')
	fig.clf()
	plt.close('all')
	del fig, axS, axS_, axT, axT_, imS, imT, imS_, imT_, ttl

# The animation function: called to produce a frame for each generation.
def animate(i):
	imT.set_data(animate.T)
	imS.set_data(animate.S)
	animate.V += np.random.rand(2,) * 0.01 - 0.005
	ttl.set_text('Iteration: %d (%d simulated second per iteration), Wind velocity = (%f, %f)'%(animate.Iter,_dt,animate.V[0],animate.V[1]))
	animate.Iter += 1
	(animate.T,animate.S) = iterate(animate.T,animate.S,animate.V)

# Bind our grid to the identifier X in the animate function's namespace.
#animate.S = S
#animate.T = T
#animate.V = V
#animate.Iter = 0
#anim = animation.FuncAnimation(fig, animate, interval=interval)

all_meas_dict = dict()
all_meas = [] # list of tuples (epoch,[meas,meas,...])
sorted_epochs = []

# Need to use a dict for insertion first because the key check is fast and easy
def addMeas(epoch,meas):
	global all_meas_dict
	if epoch not in all_meas_dict.keys():
		all_meas_dict[epoch] = []
		print "Inserting epoch " + str(epoch) + " in meas dict"
	#print "Appending measurement at epoch " + str(epoch)
	all_meas_dict[epoch].append(meas)
	
# Convert dict to list for sequential retrieval of measurements by epoch
def sortMeas():
	global sorted_epochs
	sorted_epochs = sorted(all_meas_dict.keys()) # global
	print len(all_meas_dict.keys())
	#all_meas = OrderedDict(sorted(all_meas, key=lambda m: m[0]))
	for e in sorted_epochs:
		all_meas.append((e,all_meas_dict[e]))

def getNextEpoch(currentEpoch):
	global sorted_epochs
	index = bisect.bisect_right(sorted_epochs, currentEpoch) # TODO set lo and hi to moving index
	if index < len(sorted_epochs):
		# there is more data
		print "Fetching data after epoch " + str(currentEpoch)
		return sorted_epochs[index]
	else:
		return -1
		#raise Exception("Processing has finished") # Bad bad bad don't do this in production

def getMeasForEpoch(epoch):
	global all_meas_dict
	"""
	Return array of measurements
	"""
	return all_meas_dict[epoch]
	
	


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





#sigmas = [State(T=np.ones(ny,nx)*_Ta,S=np.ones((ny,nx)),V=np.array([0.0,0.0])) for i in range(N)]

def randomBump(T,temp,ix,iy):
	# Simulate a lightning strike for initial conditions
	(ny,nx) = T.shape
	ks = 21 # kernel size = 2 * ks + 1
	#ix = np.random.randint(ks,nx-ks)
	#iy = np.random.randint(ks,ny-ks)
	T[iy-ks:iy+ks+1,ix-ks:ix+ks+1] += temp * gausskernel43x43 / np.amax(gausskernel43x43)
	
#@profile
def generateState(extents,Ta,Ta_stddev,dx,state_id,forest_fraction,ny,nx):
	# Initialize the forest grid.
	
	#S  = np.ones((ny, nx))
	#S[1:ny-1, 1:nx-1] = np.random.randint(0, 2, size=(ny-2, nx-2))
	#S[1:ny-1, 1:nx-1] = ndimage.filters.gaussian_filter(np.random.random(size=(ny-2, nx-2)) < forest_fraction,2,mode='nearest')
	S = ndimage.filters.gaussian_filter(np.clip(np.random.normal(forest_fraction,0.1,(ny,nx)),0,1),2,mode='nearest')
	# Initialize the ambient temperature grid.
	#T  = np.ones((ny, nx)) * _Ta + ndimage.filters.gaussian_filter(np.absolute(np.random.normal(0,Ta_stddev,(ny,nx))),4,mode='nearest')
	T  = np.clip(np.ones((ny, nx)) * Ta + ndimage.filters.gaussian_filter(np.random.normal(0,Ta_stddev,(ny,nx)),2,mode='nearest'),_Talclip,600)

	# add ignition point (assuming 16 points)
	sqrtN = int(math.sqrt(N))

	# Distribute evenly in rect grid over analysis area
	#igpt_x = (state_id % sqrtN + 0.5) * (nx/sqrtN) 
	#igpt_y = (state_id / sqrtN + 0.5) * (ny/sqrtN) 
	#print (igpt_x, igpt_y)
	#randomBump(T,200,igpt_x,igpt_y)

	# Distribute randomly around epicentre
	epicentre = Point2D(-1465575,-3677973)
	localec = extents.localPoint(epicentre,scale=(1.0/dx))
	# randomise
	localec.x += np.random.normal(0,20,1)
	localec.y += np.random.normal(0,20,1)

	print "Epicentre " + str(localec)
	#randomBump(T,200,localec.x,localec.y)

	#V = np.array([0.0,0.0])
	V = np.clip(np.random.normal(0,Vsigma,2),-2*Vsigma, 2*Vsigma)
	#return State(T=T,S=S,V=V,Extent=extents,dx=dx,origin=extents.topLeft())
	return State(T=T,S=S,V=V,Extent=extents,dx=dx)

def fromState(s,T,S,V):
	return State(T=T,S=S,V=V,Extent=s.Extent,dx=s.dx)

def reduceAdd(li):
	return reduce(lambda x,y: x+y, li)

#@profile
def spawnSimulation(_sigmas,dt_total,dt_step,Tnoise):
	# not the first run
	#for s in sigmas:
	#	s.fx((_nextEpoch - _epoch) * 86400,Tnoise)
	for s in _sigmas:
		s.setFxParams(dt_total,dt_step,Tnoise)
	for s in _sigmas:
		s.fx()
	# Try multiprocessing
	#p = Pool(processes = cpu_count())
	#results = p.imap(doFx,sigmas)
	#del(sigmas)
	#sigmas = [s for s in results]
	# Or try threads
	#tp = ThreadPool(len(_sigmas))
	#tp.map(doFx,_sigmas)
	#tp.wait_completion()
	#del(tp)

def mainloop():
	
	# set first epoch to max
	_epoch = 999999999999
	
	# min/max E, min/max N
	#test_extents = Rect2D(-1487906,-1480756,-3683126,-3676644)
	# one pixel below
	#test_extents = Rect2D(-1487602,-1485180,-3679064,-3676643)
	# start area small
	#test_extents = Rect2D(-1469821,-1464277,-3680275,-3676770)
	# start area  larger
	test_extents = Rect2D(-1473000,-1464277,-3683000,-3675000)
	# Forest size (number of cells in x and y directions).
	(nx, ny) = test_extents.shape() #1200, 1200
	nx /= _dx
	ny /= _dx
	
	gausskernel43x43 = np.loadtxt('gk43.txt')
	
	dr='/g/data/r78/lsd547/H8/WA/2016/01/'
	hs_h8_file='WA_jan2016_H8_ALBERS.csv'
	hs_MODIS_file='WA_jan2016_MODIS_VIIRS_ALBERS.csv'
	
	
	# define how the extents map to a local grid for analysis
	# Source grid is in Albers in metres (ignore distortions) and destination is
	# a 2D grid with 5x5 metre cell size
	gridtrans = LocalTransformation2D(test_extents,(1/_dt))
	
	H8_B07 = []
	H8_NBR = []
	
	#ymd = '20160106'
	y = '2016'
	m = '01'
	#days = (6,7,8)
	days = (6,7)
	
	# Load band 7 imagery
	for d in days:
		for hour in range(24):
			for tenminute in range(6):
				minute = tenminute * 10
				
				#fn = '%s%s%02d_%02d%02d_B07_Aus.tif'%(y,m,d,hour,minute)
				fn = '%s%s%02d_%02d%02d_B14_Aus.tif'%(y,m,d,hour,minute)
				#print 'opening ' + dr + fn
				try:
					r = gdal.Open('%s/%02d/%s'%(dr,d,fn))
					# TODO crop to test extents
					a = np.array(r.GetRasterBand(1).ReadAsArray())
					tr = r.GetGeoTransform()
					#print 'loaded'
					H8_B07.append( (utc2num('%s-%s-%02dT%02d:%02d:00Z'%(y,m,d,hour,minute)),tr,r,a) )
					#i += 1
				except AttributeError, e:
					print e
					print "Unexpected error:", sys.exc_info()[0]
					print 'File for time %02d%02d does not exist'%(hour,minute)
				fn = '%s%s%02d_%02d%02d_NBR.tif'%(y,m,d,hour,minute)
				#print 'opening ' + dr + fn
				try:
					r = gdal.Open('%s/%02d/%s'%(dr,d,fn))
					# TODO crop to test extents
					a = np.array(r.GetRasterBand(1).ReadAsArray())
					tr = r.GetGeoTransform()
					#print 'loaded'
					H8_NBR.append( (utc2num('%s-%s-%02dT%02d:%02d:00Z'%(y,m,d,hour,minute)),tr,r,a) )
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
		if not test_extents.inside(pt):
			idx.append(index)
			#print str(row[0]) + ", " + str(row[1])
		index += 1
	
	#print "Length of MODIS hotspot measurement array = " + str(hs_MODIS.shape)
	#print "Length of index list to delete MODIS hotspots = " + str(len(idx))
	hs_MODIS = np.delete(hs_MODIS, idx, axis=0)
	
	#Delete H8 entries that are outside the analysis area
	index = 0
	idx = []
	for row in hs_h8:
		pt = Point2D(row[0],row[1])
		if not test_extents.inside(pt):
			idx.append(index)
		index += 1
	
	hs_h8 = np.delete(hs_h8, idx, axis=0)
	
	print "Length of MODIS hotspot measurement array = " + str(hs_MODIS.shape)
	
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
	

	# create measurement objects
	#for row in np.nditer(hs_MODIS,axis=0):
	for row in hs_MODIS:
		# fixme make radius in grid cell counts
		#print row
		epoch = row[2]
		_epoch = min(epoch,_epoch)
		addMeas(epoch,HotspotMeasurement(Point2D(row[0],row[1]),hs_MODIS_radius,row[8],row[11],row[2]))
	
	#for row in np.nditer(hs_h8,axis=0):
	for row in hs_h8:
		# fixme make radius in grid cell counts
		epoch = row[2]
		_epoch = min(epoch,_epoch)
		cat = row[8] # TODO from category get confidence
		confidence = -1
		if cat==10: # "Processed"
			confidence = 100 # FIXME tune this std dev to a good value 
		addMeas(epoch,HotspotMeasurement(Point2D(row[0],row[1]),hs_h8_radius,row[7],confidence,row[2])) # TODO also use FRP and Fire size

	# subtract 20 minutes from _epoch to get pixels before hotspot
	_epoch -= 2 * 600.0 / 86400
	
	for e,tr,r,a in H8_B07:
		#_epoch = min(e,_epoch)
		if e >= _epoch:
			cols = r.RasterXSize
			rows = r.RasterYSize
			print "Cols = " + str(cols) + ' rows = ' + str(rows)
			print "A size = " + str(a.shape)
			rasterbounds = GetExtent(tr,cols,rows)
			#print "Raster bounds " + str(rasterbounds)
			pixSize = GetPixelSize(tr)
			# we're going from the pixSize of the raster (500m by 500m) to the grid cell size in metres (5m by 5m)
			trans = RectTransformation2D(rasterbounds,test_extents,scale=(1.0/_dx))
			# trans is local rasterbounds relative to test_extents in local grid!
			# Now it is only necessary to test local grid points to 
			# loop through each pixel and if it is in the test_extents bounds add it to the measurement list
			for col in xrange(cols):
				for row in xrange(rows):
					# Transform from raster coordinate system to grid coordinate system
					gridPixPt = trans.TransformLocalPointToLocalPoint(Point2D(col*pixSize.x,row*pixSize.y))
					#print 'Testing bounds of pixel meas at ' + str(gridPixPt.x) + ', ' + str(gridPixPt.y)
					if trans.getTargetLocalRect().inside(gridPixPt):
						# FIXME confirm that col,row referencing order is correct visually
						#print 'Adding pixel meas at ' + str(gridPixPt.x) + ', ' + str(gridPixPt.y)
						#addMeas(e,PixelTempMeasurement(gridPixPt,trans.scale*0.5,a[col][row],e))
						addMeas(e,PixelTempMeasurement(Point2D(rasterbounds.minx+col*pixSize.x,rasterbounds.miny+row*pixSize.y),pixSize.x*0.5,a[row][col],e))
			 
	#H8_B07_meas = OrderedDict(sorted(H8_B07_meas, key=lambda m: m[0]))
	
	for e,tr,r,a in H8_NBR:
		if e >= _epoch:
			cols = r.RasterXSize
			rows = r.RasterYSize
			print "NBR Cols = " + str(cols) + ' rows = ' + str(rows)
			print "NBR A size = " + str(a.shape)
			rasterbounds = GetExtent(tr,cols,rows)
			#print "Raster bounds " + str(rasterbounds)
			pixSize = GetPixelSize(tr)
			# we're going from the pixSize of the raster (500m by 500m) to the grid cell size in metres (5m by 5m)
			trans = RectTransformation2D(rasterbounds,test_extents,scale=(1.0/_dx))
			# trans is local rasterbounds relative to test_extents in local grid!
			# Now it is only necessary to test local grid points to 
			# loop through each pixel and if it is in the test_extents bounds add it to the measurement list
			for col in xrange(cols):
				for row in xrange(rows):
					# Transform from raster coordinate system to grid coordinate system
					gridPixPt = trans.TransformLocalPointToLocalPoint(Point2D(col*pixSize.x,row*pixSize.y))
					#print 'Testing bounds of pixel meas at ' + str(gridPixPt.x) + ', ' + str(gridPixPt.y)
					if trans.getTargetLocalRect().inside(gridPixPt) and np.isfinite(a).any():
						# FIXME confirm that col,row referencing order is correct visually
						#print 'Adding pixel meas at ' + str(gridPixPt.x) + ', ' + str(gridPixPt.y)
						addMeas(e,PixelNBRMeasurement(Point2D(rasterbounds.minx+col*pixSize.x,rasterbounds.miny+row*pixSize.y),pixSize.x*0.5,a[row][col],e))
	
	sortMeas()
	print "Number of epochs in sorted list = " + str(len(sorted_epochs))

	# 1) Init state with starting conditions, N = number of ensemble members (i.e. sigmas)
	_Iter = 0
	
	# 2) Create an ensemble of initted state. Do this as a for-each and MPI branch from here.
	# create ensemble members i.e. sigmas
	# First state will be noisy on purpose. Try to get best fit from the noise given the measurements.
	#sigmas = [generateState(test_extents,_Ta_stddev,_dx, i) for i in range(N)]
	sigmas = [generateState(test_extents,_Ta,_Ta_stddev,_dx,i,forest_fraction,ny,nx) for i in range(N)]
	
		
	X_mean_state = State(T=reduceAdd([s.T for s in sigmas])/N, S=reduceAdd([s.S for s in sigmas])/N, V=reduceAdd([s.V for s in sigmas])/N,Iter=_Iter)
	#makeSimPlotWindow(X_mean_state, sigmas[0])
	
	while _epoch >= 0:
		_nextEpoch = getNextEpoch(_epoch) # get first epoch
		if _epoch >= 0:
			dt_seconds = (_nextEpoch - _epoch) * 86400
			print 'Simulating for ' + str(dt_seconds) + ' seconds'
			spawnSimulation(sigmas,dt_seconds,_dt,Tnoise)
		_epoch = _nextEpoch
	
		# plot
		plotState(X_mean_state,sigmas,_epoch)
		del X_mean_state
			
		# get first lot of measurements for this epoch and check for hotspots!
		# For now don't use MPI, use a list of starting states
		
		# 3) Let x = the mean state across all ensemble members (i.e. sigma using rlabbe's pyfilter terminology)
		# inc
		_Iter += 1
		
		X_mean_state = State(T=reduceAdd([s.T for s in sigmas])/N, S=reduceAdd([s.S for s in sigmas])/N, V=reduceAdd([s.V for s in sigmas])/N,Iter=_Iter)
		# 4) let P = covariance = [sigma - x][sigma - x]^transposed, so P is NxN rank N. Use numpy.outer to computer outer product. OPTIONAL. P is huge.
		# 5) For each sigma, for each observation
		#        hx_sigma_obs = hx(sigma, observation)
		
		# get measurements for epoch
		meas_at_epoch = getMeasForEpoch(_epoch)
		
		num_meas = len(meas_at_epoch)
		HX = np.zeros((num_meas,N))
		
		for si in xrange(N):
			for mi in xrange(num_meas):
				HX[mi,si] = meas_at_epoch[mi].hx(sigmas[si])
				# also perturn half of the ensembles
				#if np.random.randint(0,2) > 0:
				meas_at_epoch[mi].perturbEnsembleState(sigmas[si])
		
		# get deviations of each ensemble member from the mean
		HA = HX - np.mean(HX,axis=0) # should work because trailing axes have same dimension i.e. columns
		
		D = np.ones((num_meas,N))
		
		for mi in xrange(num_meas):
			D[mi,:] *= meas_at_epoch[mi].d() #TODO use a fill instead
		
		Y = D - HX
		
		# R is mxm quality of observations
		R = np.zeros((num_meas,num_meas))
		for mi in xrange(num_meas):
			R[mi,mi] = meas_at_epoch[mi].r()
	
		#print "D"
		#print D
		#print "R"
		#print R
		#print "HX"
		#print HX
		#print "HA"
		#print HA
		#print "Y"
		#print Y
		
		P = R + (1.0/(N-1)) * np.matmul(HA, HA.transpose())
		
		M = np.matmul(np.linalg.inv(P), Y) # TODO make this faster via Cholesky decomposition or similar
		
		Z = np.matmul(HA.transpose(), M) * (1.0/(N-1))
		W = Z + np.identity(Z.shape[0]) # Z + I
		
		# sum all rows together
		Z_flat = np.sum(Z, axis=0)
	
		#print "Z"
		#print Z
		#print "Z_flat"
		#print Z_flat
		#print "W"
		#print W
		
		# state update
		# Xa = X + AZ/(N-1) = X + (X - mean_X)Z/(N-1) = X + XZ/(N-1) - mean_XZ/(N-1) = X(I + Z/(N-1)) - meanXZ/(N-1)
		# By making Z = HA^T M / (N-1) rewrite as
		# Xa = X + AZ = X + (X - mean_X)Z = X + XZ - mean_XZ = X(I + Z) - meanXZ = XW - meanXZ, W = I+Z
		#std_sigmas = [State(s.T - X_mean_state.T,s.S - X_mean_state.S, s.V - X_mean_state.V) for s in sigmas]
		#XbarZ = [State(T=np.multiply(-X_mean_state.T*Z_flat[i], S=np.multiply(X_mean_state.S * Z_flat[i],s.S), V=np.multiply(X_mean_state.V * Z_flat[i],s.V)) for i,s in enumerate(sigmas)]
		
		def reduceWeightSum(li,weights):
			#return reduce(lambda x,y: x+y, li)	
			# do in-place weighted sum
			if len(li) > 0:
				out = np.zeros(li[0].shape)
				for i,m in enumerate(li):
					out += weights[i] * m
				return out
			else:
				return 0 # or throw an exception
		
		#reduce add X_W
		sigmas = [fromState(s,T=reduceWeightSum([s.T for s in sigmas],W[:,j])-X_mean_state.T*Z_flat[j],S=reduceWeightSum([s.S for s in sigmas],W[:,j])-X_mean_state.S*Z_flat[j],V=reduceWeightSum([s.V for s in sigmas],W[:,j])-X_mean_state.V*Z_flat[j]) for j in xrange(N)]

		# clean up
		del HA, HX, Z, P, W, Z_flat, D, R

		# swap
		#sigmas = sigmas_X_W
		
		
	# 6) z_mean = average hx_sigma_obs over all sigmas for each observation (e.g. per-pixel average over ensemble)
	# 7) Update Pzz and Pxz... then both together produce Kalman Gain = Pxz * inv(Pzz)
	# 8) For each sigma, update via sigma += KalmanGain * [z_mean - hx + Vr] wher Vr is the perturbation added to the sigma
	# 9) update mean by summing all sigmas together (T,S,V) and dividing by N. Pointwise, cheap and easy.
	# 10) Update covariance P by P -= KalmanGain*Pzz*KalmanGain^transpose
	
if __name__ == "__main__":
	mainloop()	
