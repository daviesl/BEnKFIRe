import datacube
import xarray as xr
from datacube.storage import masking
from datacube.storage.masking import mask_to_dict
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
import matplotlib.dates
import json
import pandas as pd
from IPython.display import display
import ipywidgets as widgets
import fiona
import shapely
import shapely.geometry
from shapely.geometry import shape
import rasterio
import pickle
#import warnings
#warnings.filterwarnings('ignore')

dc = datacube.Datacube(app='dc-example')
#dc
#print dc.list_measurements()

def geom_query(geom, geom_crs='EPSG:4326'):
    """
    Create datacube query snippet for geometry
    """
    return {
        'x': (geom.bounds[0], geom.bounds[2]),
        'y': (geom.bounds[1], geom.bounds[3]),
        'crs': geom_crs
    }

def warp_geometry(geom, crs_crs, dst_crs):
    """
    warp geometry from crs_crs to dst_crs
    """
    return shapely.geometry.shape(rasterio.warp.transform_geom(crs_crs, dst_crs, shapely.geometry.mapping(geom)))

def transect(data, geom, resolution, method='nearest', tolerance=None):
    """
    
    """
    dist = [i for i in range(0, int(geom.length), resolution)]
    points = zip(*[geom.interpolate(d).coords[0] for d in dist])
    indexers = {
        data.crs.dimensions[0]: list(points[1]),
        data.crs.dimensions[1]: list(points[0])        
    }
    return data.sel_points(xr.DataArray(dist, name='distance', dims=['distance']),
                           method=method,
                           tolerance=tolerance,
                           **indexers)

def pq_fuser(dest, src):
    global valid_bit
    valid_val = (1 << valid_bit)

    no_data_dest_mask = ~(dest & valid_val).astype(bool)
    np.copyto(dest, src, where=no_data_dest_mask)

    both_data_mask = (valid_val & dest & src).astype(bool)
    np.copyto(dest, src & dest, where=both_data_mask)


query = {}
#query.update(geom_query(geom)) #comment this out if not using a polygon

#If not using a polygon/polyline, enter lat/lon here manually
#lat_max = -13.98
#lat_min = -14.05
#lon_max = 131.10
#lon_min = 130.95
#query['x'] = (lon_min, lon_max)
#query['y'] = (lat_max, lat_min)
#query['crs'] = 'EPSG:4326'

def setQueryEpoch(s,e):
	global start_of_epoch
	global end_of_epoch
	start_of_epoch = s
	end_of_epoch = e
	query = {
	    'time': (s, e),
	}

def setQueryExtent(_lat_max,_lat_min,_lon_max,_lon_min,cs):
	global query
	global start_of_epoch
	global end_of_epoch
	query['x'] = (_lon_min, _lon_max)
	query['y'] = (_lat_max, _lat_min)
	query['crs'] = 'EPSG:4326'
	#query['crs'] = 'EPSG:3577'
	#query['dask_chunks'] = {'time':1,'x':cs,'y':cs}
	print "Using query:"
	print query
	
setQueryExtent(-13.98,-14.05,131.10,130.95,1000)
#setQueryEpoch('2000-01-01','2016-07-31')
setQueryEpoch('2015-01-01','2016-10-31')

def loadBands(usels8=True,usels7=True,usels5=False):
	global time_sorted
	global nbar_clean
	global all_nbr_sorted
	global query
	global start_of_epoch
	global end_of_epoch
	global valid_bit
	
	#Define temporal range
	#start_of_epoch = '2000-01-01'
	#need a variable here that defines a rolling 'latest observation'
	#end_of_epoch =  '2016-07-31'
	
	#Define wavelengths/bands of interest, remove this kwarg to retrieve all bands
	bands_of_interest = [#'blue',
	                     #'green',
	                     'red', 
	                     'nir',
	                     'swir1', 
	                     'swir2'
	                     ]
	
	#Define sensors of interest
	sensor1 = 'ls8'
	sensor2 = 'ls7'
	sensor3 = 'ls5'
	
	nbar_list = []
	nbr_srt_list = []
	
	#Group PQ by solar day to avoid idiosyncracies of N/S overlap differences in PQ algorithm performance
	pq_albers_product = dc.index.products.get_by_name(sensor1+'_pq_albers')
	valid_bit = pq_albers_product.measurements['pixelquality']['flags_definition']['contiguous']['bits']
	
	#load sensor specific band adjustment tuples for TSS 
	ls5_tss_constant = 3983
	ls5_tss_exponent = 1.6246
	ls7_tss_constant = 3983
	ls7_tss_exponent = 1.6246
	ls8_tss_constant = 3957
	ls8_tss_exponent = 1.6436
	
	if usels8:
		#Retrieve the NBAR and PQ data for sensor n
		sensor1_nbar = dc.load(product= sensor1+'_nbar_albers', group_by='solar_day', measurements = bands_of_interest,  **query)
		sensor1_pq = dc.load(product= sensor1+'_pq_albers', group_by='solar_day', fuse_func=pq_fuser,  **query)
		
		#sensor1_nbar
		print sensor1_nbar.__dict__
		affine = sensor1_nbar.affine
		
		#This line exists to make sure that there's a 1:1 match between NBAR and PQ
		sensor1_nbar = sensor1_nbar.sel(time = sensor1_pq.time)
		
		#Generate PQ masks and apply those masks to remove cloud, cloud shadow, saturated observations
		#Generate PQ masks and apply those masks to remove cloud, cloud shadow, saturated observations
		s1_cloud_free = masking.make_mask(sensor1_pq, ga_good_pixel= True)
		s1_good_data = s1_cloud_free.pixelquality.loc[start_of_epoch:end_of_epoch]
		sensor1_nbar = sensor1_nbar.where(s1_good_data)
		
		#Fix the TSS coefficients for each sensor
		all_indices = [#'BRIGHT','GREEN','WET',
		               'NDVI','NBR','NDWI','TSS']
		sensor1_rsindex = {}
		for i, name in enumerate(all_indices):
		    #sensor1_rsindex['BRIGHT'] = pd.DataFrame((s1[0]*0.3037)+(s1[1]*0.2793)+(s1[2]*0.4343)+(s1[3]*0.5585)+(s1[4]*0.5082)+(s1[0]*0.1863))
		    #sensor1_rsindex['GREEN'] = pd.DataFrame((s1[0]*-0.2848)+(s1[1]*-0.2435)+(s1[2]*-0.5436)+(s1[3]*0.7243)+(s1[4]*0.0840)+(s1[0]*-0.1800))
		    #sensor1_rsindex['WET'] = pd.DataFrame((s1[0]*0.1509)+(s1[1]*0.1793)+(s1[2]*0.3299)+(s1[3]*0.3406)+(s1[4]*-0.7112)+(s1[0]*-0.4572))
		    sensor1_rsindex['NDVI'] = ((sensor1_nbar['nir']-sensor1_nbar['red'])/(sensor1_nbar['nir']+sensor1_nbar['red']))
		    #sensor1_rsindex['NDWI'] = ((sensor1_nbar['swir1']-sensor1_nbar['green'])/(sensor1_nbar['swir1']+sensor1_nbar['green']))
		    #sensor1_rsindex['NBR'] = ((sensor1_nbar['nir']-sensor1_nbar['swir2'])/(sensor1_nbar['nir']+sensor1_nbar['swir2']))
		    #Need this to reference into a tuple - Check with Damien    
		    #sensor1_rsindex['TSS'] = (ls8_tss_constant*((sensor1_nbar['green']+sensor1_nbar['red'])/20000)**ls8_tss_exponent)
		nbar_list.append(sensor1_nbar)
		nbr_srt_list.append(sensor1_rsindex['NDVI'])
	
	if usels7:    
		sensor2_nbar = dc.load(product= sensor2+'_nbar_albers', group_by='solar_day', measurements = bands_of_interest,  **query)
		sensor2_pq = dc.load(product= sensor2+'_pq_albers', group_by='solar_day', fuse_func=pq_fuser, **query)
		
		sensor2_nbar = sensor2_nbar.sel(time = sensor2_pq.time)
		
		s2_cloud_free = masking.make_mask(sensor2_pq, ga_good_pixel= True)
		s2_good_data = s2_cloud_free.pixelquality.loc[start_of_epoch:end_of_epoch]
		sensor2_nbar = sensor2_nbar.where(s2_good_data)
		
		all_indices = [#'BRIGHT','GREEN','WET',
		               'NDVI','NBR', 'NDWI','TSS']
		sensor2_rsindex = {}
		for i, name in enumerate(all_indices):
		    #sensor2_rsindex['BRIGHT'] = pd.DataFrame((s1[0]*0.3037)+(s1[1]*0.2793)+(s1[2]*0.4343)+(s1[3]*0.5585)+(s1[4]*0.5082)+(s1[0]*0.1863))
		    #sensor2_rsindex['GREEN'] = pd.DataFrame((s1[0]*-0.2848)+(s1[1]*-0.2435)+(s1[2]*-0.5436)+(s1[3]*0.7243)+(s1[4]*0.0840)+(s1[0]*-0.1800))
		    #sensor2_rsindex['WET'] = pd.DataFrame((s1[0]*0.1509)+(s1[1]*0.1793)+(s1[2]*0.3299)+(s1[3]*0.3406)+(s1[4]*-0.7112)+(s1[0]*-0.4572))
		    sensor2_rsindex['NDVI'] = ((sensor2_nbar['nir']-sensor2_nbar['red'])/(sensor2_nbar['nir']+sensor2_nbar['red']))
		    #sensor2_rsindex['NDWI'] = ((sensor2_nbar['swir1']-sensor2_nbar['green'])/(sensor2_nbar['swir1']+sensor2_nbar['green']))
		    #sensor2_rsindex['NBR'] = ((sensor2_nbar['nir']-sensor2_nbar['swir2'])/(sensor2_nbar['nir']+sensor2_nbar['swir2']))
		    #sensor2_rsindex['TSS'] = (ls7_tss_constant*((sensor2_nbar['green']+sensor2_nbar['red'])/20000)**ls7_tss_exponent)
		nbar_list.append(sensor2_nbar)
		nbr_srt_list.append(sensor2_rsindex['NDVI'])
		 
	# ls5 doesn't exist for 2016   
	if usels5:
		sensor3_nbar = dc.load(product= sensor3+'_nbar_albers', group_by='solar_day', measurements = bands_of_interest,   **query)
		sensor3_pq = dc.load(product= sensor3+'_pq_albers', group_by='solar_day', fuse_func=pq_fuser, **query)
		sensor3_nbar = sensor3_nbar.sel(time = sensor3_pq.time)
		s3_cloud_free = masking.make_mask(sensor3_pq, ga_good_pixel= True)
		s3_good_data = s3_cloud_free.pixelquality.loc[start_of_epoch:end_of_epoch]
		# check if data actually exists for ls5 TODO
		sensor3_nbar = sensor3_nbar.where(s3_good_data)
	
		all_indices = [#'BRIGHT','GREEN','WET',
	               'NDVI','NBR', 'NDWI','TSS']
		sensor3_rsindex = {}
		for i, name in enumerate(all_indices):
		    #sensor2_rsindex['BRIGHT'] = pd.DataFrame((s1[0]*0.3037)+(s1[1]*0.2793)+(s1[2]*0.4343)+(s1[3]*0.5585)+(s1[4]*0.5082)+(s1[0]*0.1863))
		    #sensor2_rsindex['GREEN'] = pd.DataFrame((s1[0]*-0.2848)+(s1[1]*-0.2435)+(s1[2]*-0.5436)+(s1[3]*0.7243)+(s1[4]*0.0840)+(s1[0]*-0.1800))
		    #sensor2_rsindex['WET'] = pd.DataFrame((s1[0]*0.1509)+(s1[1]*0.1793)+(s1[2]*0.3299)+(s1[3]*0.3406)+(s1[4]*-0.7112)+(s1[0]*-0.4572))
		    sensor3_rsindex['NDVI'] = ((sensor3_nbar['nir']-sensor3_nbar['red'])/(sensor3_nbar['nir']+sensor3_nbar['red']))
		    #sensor3_rsindex['NDWI'] = ((sensor3_nbar['swir1']-sensor3_nbar['green'])/(sensor3_nbar['swir1']+sensor3_nbar['green']))
		    #sensor3_rsindex['NBR'] = ((sensor3_nbar['nir']-sensor3_nbar['swir2'])/(sensor3_nbar['nir']+sensor3_nbar['swir2']))
		    #sensor3_rsindex['TSS'] = ((sensor3_nbar['green']+sensor3_nbar['red'])/2)
		    #sensor3_rsindex['TSS'] = (ls5_tss_constant*((sensor3_nbar['green']+sensor3_nbar['red'])/20000)**ls5_tss_exponent)
		
		nbar_list.append(sensor3_nbar)
		nbr_srt_list.append(sensor3_rsindex['NDVI'])
	    
	#Concatenate and sort the different sensor xarrays into a single xarray
	
	nbar_clean = xr.concat(nbar_list, dim='time')
	#nbar_clean = xr.concat([sensor1_nbar, sensor2_nbar], dim='time')
	time_sorted = nbar_clean.time.argsort()
	nbar_clean = nbar_clean.isel(time=time_sorted)
	nbar_clean.attrs['affine'] = affine
	
	'''
	all_tss_sorted = xr.concat([sensor1_rsindex['TSS'], sensor2_rsindex['TSS'], sensor3_rsindex['TSS']], dim='time')
	time_sorted = all_tss_sorted.time.argsort()
	all_tss_sorted = all_tss_sorted.isel(time=time_sorted)'''
	
	"""all_ndvi_sorted = xr.concat([sensor1_rsindex['NDVI'], sensor2_rsindex['NDVI'], sensor3_rsindex['NDVI']], dim='time')
	time_sorted = all_ndvi_sorted.time.argsort()
	all_ndvi_sorted = all_ndvi_sorted.isel(time=time_sorted)
	"""
	#all_nbr_sorted = xr.concat([sensor1_rsindex['NBR'], sensor2_rsindex['NBR'], sensor3_rsindex['NBR']], dim='time')
	all_nbr_sorted = xr.concat(nbr_srt_list,  dim='time')
	time_sorted = all_nbr_sorted.time.argsort()
	all_nbr_sorted = all_nbr_sorted.isel(time=time_sorted)
	all_nbr_sorted.attrs['affine'] = affine
	
	#clean up per sensor xarrays to free up some memory
	#del sensor1_nbar
	#del sensor2_nbar
	#del sensor3_nbar
	#del sensor1_rsindex
	#del sensor2_rsindex
	#del sensor3_rsindex
	
	print 'The number of time slices at this location is' 
	print all_nbr_sorted.shape[0]

def plotTimeSlice(time_slice):
	#select time slice of interest
	#time_slice = 287
	rgb = nbar_clean.isel(time =time_slice).to_array(dim='color').sel(color=['swir2', 'nir', 'green']).transpose('y', 'x', 'color')
	fake_saturation = 3500
	clipped_visible = rgb.where(rgb<fake_saturation).fillna(fake_saturation)
	max_val = clipped_visible.max(['y', 'x'])
	scaled = (clipped_visible / max_val)
	fig = plt.figure()
	gs = gridspec.GridSpec(1,1, width_ratios=[1,1])
	ax2 = fig.add_subplot(gs[0,0])
	ax2.set_title(time_slice)
	ax2.imshow(scaled, interpolation = 'nearest')
	plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
	plt.subplots_adjust(wspace=0.1, hspace=0.1)
	plt.show()
	
def getNBRExtent():
	global nbar_clean
	return [nbar_clean.coords['x'].max(),nbar_clean.coords['x'].min(),nbar_clean.coords['y'].max(),nbar_clean.coords['y'].min()]

def plotNBR(visext,fn='landsat_ndvi.txt',i=0,vmin=0.85,vmax=1):
	#annual_nbr = all_nbr_sorted.groupby('time.year')
	annual_nbr = all_nbr_sorted
	
	#annual_nbrmin = annual_nbr.min(dim = 'time')
	#annual_nbrmax = annual_nbr.max(dim = 'time')
	#dnbr = annual_nbrmax - annual_nbrmin
	dnbr = annual_nbr.median(dim = 'time')
	np.savetxt(fn,dnbr)
	print 'extents:'
	print visext
	fig = plt.figure()
	gs = gridspec.GridSpec(1,1, width_ratios=[1,1])
	
	ax1 = fig.add_subplot(gs[0,0])
	
	#i = 15
	#vmin = 0.85
	#vmax = 1
	
	#ax1.imshow(dnbr[i], interpolation = 'nearest', cmap = 'RdYlGn_r', vmin = vmin, vmax = vmax, 
	#           extent=visext)
	ax1.imshow(dnbr, interpolation = 'nearest', cmap = 'RdYlGn_r', vmin = vmin, vmax = vmax, 
	           extent=visext)
	#ax1.set_title(dnbr.year[i])
	ax1.set_title("Median NDVI")
	#savefig('/home/554/lxl554/Desktop/NBR graphics/fishriver_deltaNBR_2009.png')
	plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
	plt.subplots_adjust(wspace=0.1, hspace=0.1)
	plt.show()

