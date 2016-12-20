import pyximport
#from functools import partial
#import multiprocessing as mp



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
import pickle

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

#_T_stddev = 20 # degrees (not used afaik)
_Ta_stddev = 80
Tnoise = 5.0 # per ensemble member noise added at start of each simulation run

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
	def setFxParams(self,file_id,dt,step_dt,Tnoise):
		self.file_id = file_id
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
		#print "Perturbing temperature." # Bounds shape: " + str(b) + " Kernel shape: " + str(p)
		self.T[b.minyi():b.maxyi(),b.minxi():b.maxxi()] += p
	def perturbTempMax(self,b,p):
		#print "Perturbing temperature." # Bounds shape: " + str(b) + " Kernel shape: " + str(p)
		self.T[b.minyi():b.maxyi(),b.minxi():b.maxxi()] = np.maximum(p,self.T[b.minyi():b.maxyi(),b.minxi():b.maxxi()])
	def perturbFuel(self,b,p):
		#print "Perturbing fuel." # Bounds shape: " + str(b) + " Kernel shape: " + str(p)
		self.S[b.minyi():b.maxyi(),b.minxi():b.maxxi()] += p
	@staticmethod
	def randomBump(T,temp,ix,iy):
		# Simulate a lightning strike for initial conditions
		(ny,nx) = T.shape
		ks = 21 # kernel size = 2 * ks + 1
		#ix = np.random.randint(ks,nx-ks)
		#iy = np.random.randint(ks,ny-ks)
		T[iy-ks:iy+ks+1,ix-ks:ix+ks+1] += temp * gausskernel43x43 / np.amax(gausskernel43x43)
	#@profile
	@classmethod
	def generateState(cls,extents,Ta,Ta_stddev,dx,state_id,N,forest_fraction,ny,nx):
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
		#cls.randomBump(T,200,igpt_x,igpt_y)
	
		# Distribute randomly around epicentre
		epicentre = Point2D(-1465575,-3677973)
		localec = extents.localPoint(epicentre,scale=(1.0/dx))
		# randomise
		localec.x += np.random.normal(0,20,1)
		localec.y += np.random.normal(0,20,1)
	
		print "Epicentre " + str(localec)
		#cls.randomBump(T,200,localec.x,localec.y)
	
		#V = np.array([0.0,0.0])
		V = np.clip(np.random.normal(0,Vsigma,2),-2*Vsigma, 2*Vsigma)
		#return State(T=T,S=S,V=V,Extent=extents,dx=dx,origin=extents.topLeft())
		return cls(T=T,S=S,V=V,Extent=extents,dx=dx)

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
			self.temp = 600 # ignition?
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
		localBounds = Rect2D.fromCircle(state.Extent.localPoint(self.centre,scale=abs(1.0/state.dx)),self.radius/state.dx+1)
		return localBounds.clipToInt(state.LocalExtent())
	def perturbEnsembleState(self,state):
		lb = self.getLocalBounds(state)
		(cols,rows) = lb.shape()
		cols = int(cols)
		rows = int(rows)
		border = 1
		if cols/2 > border and rows/2 > border:
			rx = np.random.randint(0+border,cols - border)
			ry = np.random.randint(0+border,rows - border)
			gk_radius = 4
			#gk = np.random.normal(self.d()-hx_PixelTemp(state.T,lb),self.r(),1) * gkern2(rows,cols,ry,rx,nsig=gk_radius) * gk_radius * 2 * math.pi
			gk = np.random.normal(self.d(),self.r(),1) * gkern2(rows,cols,ry,rx,nsig=gk_radius) * gk_radius * 2 * math.pi
			print "Hotspot perturbation by " + str(np.max(gk)) + " degrees"
			state.perturbTempMax(lb,gk)

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
		localBounds = Rect2D.fromCircle(state.Extent.localPoint(self.centre,scale=abs(1.0/state.dx)),abs(self.radius/state.dx)+1)
		return localBounds.clipToInt(state.LocalExtent())
	def perturbEnsembleState(self,state):
		lb = self.getLocalBounds(state)
		(cols,rows) = lb.shape()
		cols = int(cols)
		rows = int(rows)
		gn_radius = 5
		variance = self.r() * gn_radius * 2 * math.pi
		state.perturbTemp(lb,ndimage.filters.gaussian_filter(np.random.normal(self.d()-self.hx(state),variance,(rows,cols)),gn_radius,mode='nearest'))

def sinScale(v):
	return math.sin(np.clip(v,0,1)* math.pi / 2.0)

class PixelNBRMeasurement(Measurement):
	def __init__(self,centre,radius,nbr,epoch):
		self.nbr = nbr
		self.centre = centre
		self.radius = radius
		self.epoch = epoch
		self.quality = 0.2 # 20% std dev at a guess
	def hx(self,state):
		"""
		return computed average fuel load from within bounds or mask
		"""
		localBounds = self.getLocalBounds(state)
		# get nbr from state.S within localBounds
		#return math.sin(hx_PixelNBR(state.S,localBounds) * math.pi / 2.0)
		return sinScale(sinScale(hx_PixelNBR(state.S,localBounds) * 0.5 + 0.5))
	def d(self):
		return self.nbr
	def r(self):
		return self.quality
	def getLocalBounds(self,state):
		# clip to local extents
		localBounds = Rect2D.fromCircle(state.Extent.localPoint(self.centre,scale=abs(1.0/state.dx)),abs(self.radius/state.dx)+1)
		return localBounds.clipToInt(state.LocalExtent())
	def perturbEnsembleState(self,state):
		lb = self.getLocalBounds(state)
		(cols,rows) = lb.shape()
		cols = int(cols)
		rows = int(rows)
		gn_radius = 5
		variance = self.r() * gn_radius * 2 * math.pi
		state.perturbFuel(lb,ndimage.filters.gaussian_filter(np.random.normal(self.d()-self.hx(state),variance,(rows,cols)),gn_radius,mode='nearest'))
