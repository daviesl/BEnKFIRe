import dcnbr as dcnbr

#dcnbr.setQueryExtent(-13.98,-14.05,131.10,130.95)
#setQueryEpoch('2000-01-01','2016-07-31')
#dcnbr.setQueryEpoch('2015-01-01','2016-10-31')

#area='waroona' #'tas'
area='tas'
fn = '/local/r78/'

if area=='tas':
	print 'Setting extents for Tasmania'
	dcnbr.setQueryExtent(-40.8,-41.5,145.5,144.6,1000)
	print 'Setting epoch'
	dcnbr.setQueryEpoch('2015-08-01','2016-09-30')
	fn += 'tas_landsat_dnbr.txt'
elif area=='waroona':
	print 'Setting extents for Waroona'
	dcnbr.setQueryExtent(-32.7,-33.1,116.4,115.5,1000)
	print 'Setting epoch'
	dcnbr.setQueryEpoch('2015-11-01','2016-02-28')
	fn += 'waroona_landsat_dnbr.txt'

print 'loading bands'
dcnbr.loadBands()
print 'plotting'
dcnbr.plotNBR(dcnbr.getNBRExtent(),fn=fn,vmin=0.25)
