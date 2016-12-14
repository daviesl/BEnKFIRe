import dcnbr

#dcnbr.setQueryExtent(-32.8,-33.1,116.21,115.66)
#dcnbr.setQueryExtent(-32.808,-33.092,116.21,115.717,100)
dcnbr.setQueryExtent(-32.808,-33.092,116.21,115.817,100)
dcnbr.setQueryEpoch('2015-12-25','2016-01-31')
dcnbr.loadBands(True,False,False)
extent = dcnbr.getNBRExtent()
#dcnbr.plotTimeSlice(0)
dcnbr.plotNBR(extent,vmin=0,vmax=1)
