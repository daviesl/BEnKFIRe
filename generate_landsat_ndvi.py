import dcndvi
import sys

#dcndvi.setQueryExtent(-13.98,-14.05,131.10,130.95)
#setQueryEpoch('2000-01-01','2016-07-31')
#dcndvi.setQueryEpoch('2015-01-01','2016-10-31')

area='waroona' #'tas'
#area='sa_regional2'
fn = '/local/r78/'

if area=='tas':
	print 'Setting extents for Tasmania'
	dcndvi.setQueryExtent(-40.8,-41.5,145.5,144.6,1000)
	print 'Setting epoch'
	dcndvi.setQueryEpoch('2015-08-01','2016-09-30')
	fn += 'tas_landsat_dnbr.txt'
	print 'loading bands'
	dcndvi.loadBands()
	print 'plotting'
	dcndvi.plotNBR(dcndvi.getNBRExtent(),fn=fn,vmin=0.25)
elif area=='waroona':
	print 'Setting extents for Waroona'
	#dcndvi.setQueryExtent(-32.7,-33.1,116.4,115.5,1000)
	dcndvi.setQueryExtent(-32.83,-32.98,116.22,116.09,1000)
	#dcndvi.setQueryExtent(-1464277,-1473000,-3675000,-3683000,1000)
	print 'Setting epoch'
	dcndvi.setQueryEpoch('2015-10-01','2015-12-31') # pre fire
	#dcndvi.setQueryEpoch('2016-01-07','2016-03-31') # post fire
	fn += 'waroona_landsat_ndvi_pre_fire.txt'
	print 'loading bands'
	dcndvi.loadBands()
	print 'plotting'
	dcndvi.plotNBR(dcndvi.getNBRExtent(),fn=fn,vmin=0.00)
elif area=='sa_regional2':
	print 'Setting extents for SA Regional Subset'
	dcndvi.setQueryExtent(-29.635,-30.857,139.499,138.002,2000)
	print 'Setting epoch'
	dcndvi.setQueryEpoch('2015-010-25','2015-12-30')
	fn += 'sa_regional_subset_landsat_dnbr.txt'
	print 'loading bands'
	dcndvi.loadBands()
	print 'plotting'
	dcndvi.plotNBR(dcndvi.getNBRExtent(),fn=fn,vmin=0.25)
elif area=='sa_regional':
	print 'Setting extents for SA Regional 1'
	#dcndvi.setQueryExtent(-28.7,-32.4,142.3,137.4,1000)
	dcndvi.setQueryExtent(-28.7,-30.6,139.9,137.4,2000)
	print 'Setting epoch 1'
	dcndvi.setQueryEpoch('2015-010-01','2016-01-17')
	fn += 'sa_regional_landsat_dnbr_1.txt'
	print 'loading bands 1'
	dcndvi.loadBands()
	print 'plotting 1'
	dcndvi.plotNBR(dcndvi.getNBRExtent(),fn=fn,vmin=0.25)
	print 'Setting extents for SA Regional 2'
	dcndvi.setQueryExtent(-28.7,-30.6,142.3,139.8,2000)
	print 'Setting epoch 2'
	dcndvi.setQueryEpoch('2015-010-01','2016-01-17')
	fn += 'sa_regional_landsat_dnbr_2.txt'
	print 'loading bands 2'
	dcndvi.loadBands()
	print 'plotting 2'
	dcndvi.plotNBR(dcndvi.getNBRExtent(),fn=fn,vmin=0.25)
	print 'Setting extents for SA Regional 3'
	dcndvi.setQueryExtent(-30.5,-32.4,139.9,137.4,2000)
	print 'Setting epoch 3'
	dcndvi.setQueryEpoch('2015-010-01','2016-01-17')
	fn += 'sa_regional_landsat_dnbr_3.txt'
	print 'loading bands 3'
	dcndvi.loadBands()
	print 'plotting 3'
	dcndvi.plotNBR(dcndvi.getNBRExtent(),fn=fn,vmin=0.25)
	print 'Setting extents for SA Regional 4'
	dcndvi.setQueryExtent(-30.5,-32.4,142.3,139.8,2000)
	print 'Setting epoch 4'
	dcndvi.setQueryEpoch('2015-010-01','2016-01-17')
	fn += 'sa_regional_landsat_dnbr_4.txt'
	print 'loading bands 4'
	dcndvi.loadBands()
	print 'plotting 4'
	dcndvi.plotNBR(dcndvi.getNBRExtent(),fn=fn,vmin=0.25)
else:
	print 'Error: invalid region specified'
	sys.exit(0)

