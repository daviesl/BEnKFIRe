import pyximport
#from functools import partial
#import multiprocessing as mp
from multiprocessing import Pool, cpu_count
#from multiprocessing.pool import ApplyResult
#from copy_reg import pickle
#from types import MethodType

import pickle

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
import os
#from filterpy.kalman import EnsembleKalmanFilter as EnKF
import bisect

import wf_state as WF

IS_PY2 = sys.version_info < (3, 0)

if IS_PY2:
    from Queue import Queue
else:
    from queue import Queue

from threading import Thread

#def _pickle_method(method):
#	func_name = method.im_func.__name__
#	obj = method.im_self
#	cls = method.im_class
#	return _unpickle_method, (func_name, obj, cls)
#
#def _unpickle_method(func_name, obj, cls):
#	for cls in cls.mro():
#		try:
#			func = cls.__dict__[func_name]
#		except KeyError:
#			pass
#		else:
#			break
#	return func.__get__(obj, cls)


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


# pickle line required for multiprocessing
#pickle(MethodType, _pickle_method, _unpickle_method)


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
	imT = axT.imshow(T, cmap='jet', vmin=WF._Tal, vmax=1000, interpolation='none')#, interpolation='nearest')
	imS0 = axS0.imshow(s0.S, cmap='gray', vmin=0, vmax=1, interpolation='none')#, interpolation='nearest')
	imT0 = axT0.imshow(s0.T, cmap='jet', vmin=WF._Tal, vmax=1000, interpolation='none')#, interpolation='nearest')
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
	fig = plt.figure(figsize=(12, 12),dpi=72)
	N_ = len(sigmas_)
	sqrtN_ = int(math.sqrt(N_))
	gs = gridspec.GridSpec(sqrtN_ + 1,2*sqrtN_) #, width_ratios=[1,1])
	# FIXME remove animation and write images to disk if desired
	axS = fig.add_subplot(gs[0,0:sqrtN_])
	axS.set_axis_off()
	axT = fig.add_subplot(gs[0,sqrtN_:(2*sqrtN_)])
	axT.set_axis_off()
	imS = axS.imshow(s.S, cmap='gray', vmin=0, vmax=1, interpolation='none')#, interpolation='nearest')
	imT = axT.imshow(s.T, cmap='jet', vmin=WF._Tal, vmax=1000, interpolation='none')#, interpolation='nearest')
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
		imT_.append(axT_[i].imshow(s_.T, cmap='jet', vmin=WF._Tal, vmax=1000, interpolation='none'))#, interpolation='nearest')
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
	#print len(all_meas_dict.keys())
	#all_meas = OrderedDict(sorted(all_meas, key=lambda m: m[0]))
	#for e in sorted_epochs:
	#	all_meas.append((e,all_meas_dict[e]))

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





def fromState(s,T,S,V):
	return WF.State(T=T,S=S,V=V,Extent=s.Extent,dx=s.dx)

def reduceAdd(li):
	return reduce(lambda x,y: x+y, li)

from subprocess import call
def doFx(s):
	#s.fx()
	#return s
	spath = '/g/data/r78/lsd547/scratch/state_' + str(s.file_id)
	with open(spath, 'wb') as f:
		p = pickle.Pickler(f,protocol=pickle.HIGHEST_PROTOCOL)
		p.dump(s)
		del(p)
	#del(s)
	call(["./sim_wf.py",spath])
	with open(spath, 'rb') as f:
		up = pickle.Unpickler(f)
		s.update(up.load()) #stations
		#s = up.load()
		del(up)
	os.remove(spath)
	#return s
	

#@profile
def spawnSimulation(tp,_sigmas,dt_total,dt_step,Tnoise):
	for i,s in enumerate(_sigmas):
		s.setFxParams(i,dt_total,dt_step,Tnoise)
	#for s in _sigmas:
	#	s.fx()
	# Try multiprocessing
	#p = Pool(processes = cpu_count())
	#_sigmas = p.map(doFx,_sigmas)
	#p.close()
	#p.join()
	#results = p.imap(doFx,_sigmas)
	#for s in _sigmas:
	#	del(s)
	#for s in results:
	#	_sigmas.append(s)
	
	#sigmas = [s for s in results]
	# Or try threads
	tp.map(doFx,_sigmas)
	tp.wait_completion()

def mainloop():
	
	# set first epoch to max
	_epoch = 999999999999
	N = 16
	# min/max E, min/max N
	#test_extents = Rect2D(-1487906,-1480756,-3683126,-3676644)
	# one pixel below
	#test_extents = Rect2D(-1487602,-1485180,-3679064,-3676643)
	# start area small
	#test_extents = Rect2D(-1469821,-1464277,-3680275,-3676770)
	# start area  larger
	test_extents = WF.Rect2D(-1473000,-1464277,-3683000,-3675000)
	# Forest size (number of cells in x and y directions).
	(nx, ny) = test_extents.shape() #1200, 1200
	nx /= WF._dx
	ny /= WF._dx
	
	gausskernel43x43 = np.loadtxt('gk43.txt')
	
	dr='/g/data/r78/lsd547/H8/WA/2016/01/'
	hs_h8_file='WA_jan2016_H8_ALBERS.csv'
	hs_MODIS_file='WA_jan2016_MODIS_VIIRS_ALBERS.csv'
	
	
	# define how the extents map to a local grid for analysis
	# Source grid is in Albers in metres (ignore distortions) and destination is
	# a 2D grid with 5x5 metre cell size
	gridtrans = WF.LocalTransformation2D(test_extents,(1/WF._dt))
	
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
					H8_NBR.append( (utc2num('%s-%s-%02dT%02d:%02d:00Z'%(y,m,d,hour,minute)),tr,r,a,None) )
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
		pt = WF.Point2D(row[0],row[1])
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
		pt = WF.Point2D(row[0],row[1])
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
	hs_MODIS_radius = 0.5 * 500.0 / WF._dx
	
	#hs_h8_meas = []
	hs_h8_radius = 0.5 * 2000.0 / WF._dx
	
	#H8_B07_meas = []
	H8_B07_radius = 500.0 / WF._dx
	

	# create measurement objects
	#for row in np.nditer(hs_MODIS,axis=0):
	for row in hs_MODIS:
		# fixme make radius in grid cell counts
		#print row
		epoch = row[2]
		_epoch = min(epoch,_epoch)
		addMeas(epoch,WF.HotspotMeasurement(WF.Point2D(row[0],row[1]),hs_MODIS_radius,row[8],row[11],row[2]))
	
	#for row in np.nditer(hs_h8,axis=0):
	for row in hs_h8:
		# fixme make radius in grid cell counts
		epoch = row[2]
		_epoch = min(epoch,_epoch)
		cat = row[8] # TODO from category get confidence
		confidence = -1
		if cat==10: # "Processed"
			confidence = 100 # FIXME tune this std dev to a good value 
		addMeas(epoch,WF.HotspotMeasurement(WF.Point2D(row[0],row[1]),hs_h8_radius,row[7],confidence,row[2])) # TODO also use FRP and Fire size

	# subtract 20 minutes from _epoch to get pixels before hotspot
	_epoch -= 2 * 600.0 / 86400
	
	# sort the measurements and get the epochs of hotspots
	sortMeas()

	#H8_B07_meas = OrderedDict(sorted(H8_B07_meas, key=lambda m: m[0]))
	last_a = []
	last_e = 99999
	for i,(e,tr,r,a,dnbr) in enumerate(H8_NBR):
		if i < (len(H8_NBR) - 1) and np.isfinite(a).any():
			(next_e,next_tr,next_r,next_a,next_dnbr) = H8_NBR[i+1]
			# don't use last_e for now, but maybe use it as a threshold, i.e. if last_e < timelimit: add dnbr
			epoch_diff = next_e - e
			if epoch_diff < (1300.0 / 86400):
				# compute dnbr
				dnbr = np.clip(a-next_a,0,2)
				dnbr /= np.amax(dnbr)
				dnbr_scaled = np.zeros(dnbr.shape)
				dnbr_quality = np.ones(dnbr.shape) * 1000
				# scale by a hotspot map
				# first get the epoch for dnbr and search for hotspots within the ten minute interval minus 5 minutes.
				search_e = e - (300.0 / 86400.0)
				mfe = []
				search_e = getNextEpoch(search_e)
				#print "searching for hotspots at " + str(search_e)
				while search_e < next_e:
					mfe.extend(getMeasForEpoch(search_e))
					search_e = getNextEpoch(search_e)
				if len(mfe) > 0:
					print "Found hotspot measurmeents for DNBR"
					# get hotspot measurements
					cols = r.RasterXSize
					rows = r.RasterYSize
					rasterbounds = WF.GetExtent(tr,cols,rows)
					#print "raster bound region " + str(rasterbounds)
					localbounds = WF.Rect2D(0,cols,0,rows)
					#print "local bound region " + str(localbounds)
					pixSize = WF.GetPixelSize(tr)
					trans = WF.RectTransformation2D(rasterbounds,test_extents,scale=(1.0/WF._dx))
					num_meas = len(mfe)
					# find any hotspots, and scale the dnbr pixels to those hotspots
					# no hotspots means no pixels
					for mi in xrange(num_meas):
						scale_region_centre = rasterbounds.localPoint(mfe[mi].centre,(1.0/pixSize.x))
						#print "centre = " + str(mfe[mi].centre)
						#print "scale_centre = " + str(scale_region_centre)
						scale_region = WF.Rect2D.fromCircle(scale_region_centre,mfe[mi].radius / pixSize.x)
						#print "Scale region " + str(scale_region)
						scale_region = scale_region.clipToInt(localbounds)
						#print "Scale region " + str(scale_region)
						if scale_region.area() > 0:
							#print "Adding scaled DNBR"
							dnbr_quality[scale_region.minyi():scale_region.maxyi(),scale_region.minxi():scale_region.maxxi()] = mfe[mi].r() #fixme for overlapping hotspots
							dnbr_scaled[scale_region.minyi():scale_region.maxyi(),scale_region.minxi():scale_region.maxxi()] = dnbr[scale_region.minyi():scale_region.maxyi(),scale_region.minxi():scale_region.maxxi()] * mfe[mi].d() / np.amax(dnbr[scale_region.minyi():scale_region.maxyi(),scale_region.minxi():scale_region.maxxi()])
					for col in xrange(cols):
						for row in xrange(rows):
							# Transform from raster coordinate system to grid coordinate system
							gridPixPt = trans.TransformLocalPointToLocalPoint(WF.Point2D(col*pixSize.x,row*pixSize.y))
							if dnbr_scaled[row][col] > 0 and trans.getTargetLocalRect().inside(gridPixPt):
								print "Adding DNBR measurement, temperature " + str(dnbr_scaled[row][col])
								addMeas(e,WF.PixelDNBRMeasurement(WF.Point2D(rasterbounds.minx+(col+0.5)*pixSize.x,rasterbounds.miny+(row+0.5)*pixSize.y),pixSize.x*0.5,dnbr_scaled[row][col],dnbr_quality[row][col],e))
					
			
				#H8_NBR[i] = (e,tr,r,a,dnbr)
			
	for e,tr,r,a in H8_B07:
		#_epoch = min(e,_epoch)
		if e >= _epoch:
			cols = r.RasterXSize
			rows = r.RasterYSize
			#print "Cols = " + str(cols) + ' rows = ' + str(rows)
			#print "A size = " + str(a.shape)
			rasterbounds = WF.GetExtent(tr,cols,rows)
			#print "Raster bounds " + str(rasterbounds)
			pixSize = WF.GetPixelSize(tr)
			# we're going from the pixSize of the raster (500m by 500m) to the grid cell size in metres (5m by 5m)
			trans = WF.RectTransformation2D(rasterbounds,test_extents,scale=(1.0/WF._dx))
			# trans is local rasterbounds relative to test_extents in local grid!
			# Now it is only necessary to test local grid points to 
			# loop through each pixel and if it is in the test_extents bounds add it to the measurement list
			for col in xrange(cols):
				for row in xrange(rows):
					# Transform from raster coordinate system to grid coordinate system
					gridPixPt = trans.TransformLocalPointToLocalPoint(WF.Point2D(col*pixSize.x,row*pixSize.y))
					#print 'Testing bounds of pixel meas at ' + str(gridPixPt.x) + ', ' + str(gridPixPt.y)
					if trans.getTargetLocalRect().inside(gridPixPt):
						# FIXME confirm that col,row referencing order is correct visually
						#print 'Adding pixel meas at ' + str(gridPixPt.x) + ', ' + str(gridPixPt.y)
						#addMeas(e,PixelTempMeasurement(gridPixPt,trans.scale*0.5,a[col][row],e))
						addMeas(e,WF.PixelTempMeasurement(WF.Point2D(rasterbounds.minx+(col+0.5)*pixSize.x,rasterbounds.miny+(row+0.5)*pixSize.y),pixSize.x*0.5,a[row][col],e))
			 
	for i,(e,tr,r,a,dnbr) in enumerate(H8_NBR):
		if e >= _epoch:
			cols = r.RasterXSize
			rows = r.RasterYSize
			#print "NBR Cols = " + str(cols) + ' rows = ' + str(rows)
			#print "NBR A size = " + str(a.shape)
			rasterbounds = WF.GetExtent(tr,cols,rows)
			#print "Raster bounds " + str(rasterbounds)
			pixSize = WF.GetPixelSize(tr)
			# we're going from the pixSize of the raster (500m by 500m) to the grid cell size in metres (5m by 5m)
			trans = WF.RectTransformation2D(rasterbounds,test_extents,scale=(1.0/WF._dx))
			# trans is local rasterbounds relative to test_extents in local grid!
			# Now it is only necessary to test local grid points to 
			# loop through each pixel and if it is in the test_extents bounds add it to the measurement list
			for col in xrange(cols):
				for row in xrange(rows):
					# Transform from raster coordinate system to grid coordinate system
					gridPixPt = trans.TransformLocalPointToLocalPoint(WF.Point2D(col*pixSize.x,row*pixSize.y))
					#print 'Testing bounds of pixel meas at ' + str(gridPixPt.x) + ', ' + str(gridPixPt.y)
					if trans.getTargetLocalRect().inside(gridPixPt) and np.isfinite(a).any():
						# FIXME confirm that col,row referencing order is correct visually
						print 'Adding NBR meas at ' + str(gridPixPt.x) + ', ' + str(gridPixPt.y)
						addMeas(e,WF.PixelNBRMeasurement(WF.Point2D(rasterbounds.minx+(col+0.5)*pixSize.x,rasterbounds.miny+(row+0.5)*pixSize.y),pixSize.x*0.5,a[row][col],e))
	
	sortMeas()
	print "Number of epochs in sorted list = " + str(len(sorted_epochs))

	# 1) Init state with starting conditions, N = number of ensemble members (i.e. sigmas)
	_Iter = 0
	
	# 2) Create an ensemble of initted state. Do this as a for-each and MPI branch from here.
	# create ensemble members i.e. sigmas
	# First state will be noisy on purpose. Try to get best fit from the noise given the measurements.
	#sigmas = [State.generateState(test_extents,WF._Ta_stddev,WF._dx, i) for i in range(N)]
	sigmas = [WF.State.generateState(test_extents,WF._Ta,WF._Ta_stddev,WF._dx,i,N,WF.forest_fraction,ny,nx) for i in range(N)]
	
		
	tp = ThreadPool(len(sigmas))

	X_mean_state = WF.State(T=reduceAdd([s.T for s in sigmas])/N, S=reduceAdd([s.S for s in sigmas])/N, V=reduceAdd([s.V for s in sigmas])/N,Iter=_Iter)
	#makeSimPlotWindow(X_mean_state, sigmas[0])
	
	while _epoch >= 0:
		_nextEpoch = getNextEpoch(_epoch) # get first epoch
		del X_mean_state

		dt_seconds = (_nextEpoch - _epoch) * 86400
		_epoch = _nextEpoch
		print 'Simulating for ' + str(dt_seconds) + ' seconds'
		spawnSimulation(tp,sigmas,dt_seconds,WF._dt,WF.Tnoise)

	
			
		# get first lot of measurements for this epoch and check for hotspots!
		# For now don't use MPI, use a list of starting states
		
		# 3) Let x = the mean state across all ensemble members (i.e. sigma using rlabbe's pyfilter terminology)
		# inc
		_Iter += 1
		
		X_mean_state = WF.State(T=reduceAdd([s.T for s in sigmas])/N, S=reduceAdd([s.S for s in sigmas])/N, V=reduceAdd([s.V for s in sigmas])/N,Iter=_Iter)
		# plot
		plotState(X_mean_state,sigmas,_epoch)
		# 4) let P = covariance = [sigma - x][sigma - x]^transposed, so P is NxN rank N. Use numpy.outer to computer outer product. OPTIONAL. P is huge.
		# get measurements for epoch
		meas_at_epoch = getMeasForEpoch(_epoch)
		num_meas = len(meas_at_epoch)
		
		# 5) For each sigma, for each observation
		#        hx_sigma_obs = hx(sigma, observation)
		
		HX = np.zeros((num_meas,N))

		def rand_polar2XY(rsig):
			theta = np.random.uniform(math.pi - math.pi * 0.25,math.pi - + math.pi * 0.25,1)
			r = np.random.uniform(0,rsig,1)
			return np.array([r * np.cos(theta), r * np.sin(theta)]).flatten()
			


		for si in xrange(N):
			# perturb wind velocity
			#sigmas[si].V = np.clip(sigmas[si].V+np.random.uniform(-WF.Vsigma,WF.Vsigma,2),-2*WF.Vsigma,2*WF.Vsigma)
			print "Broadcast V " + str(sigmas[si].V)
			o = rand_polar2XY(WF.Vsigma)
			print "Broadcast o " + str(o)
			#sigmas[si].V += rand_polar2XY(WF.Vsigma)
			sigmas[si].V += o
			mag = np.linalg.norm(sigmas[si].V,ord=2)
			if mag > WF.Vsigma*2:
				sigmas[si].V *= WF.Vsigma*2.0/mag
			for mi in xrange(num_meas):
				#if np.random.randint(0,2) > 0:
				cn = meas_at_epoch[mi].__class__.__name__
				#print "Checking meas " + cn
				if cn == 'PixelDNBRMeasurement':
					print 'Perturbing ensemble'
					meas_at_epoch[mi].perturbEnsembleState(sigmas[si])
		
		for si in xrange(N):
			for mi in xrange(num_meas):
				HX[mi,si] = meas_at_epoch[mi].hx(sigmas[si])
		
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
