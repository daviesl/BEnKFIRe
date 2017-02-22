import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn import metrics
from matplotlib.dates import datestr2num, num2date
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
import sys
import math


# Globals to use in classifier
class FireRegion(object):
	def __init__(self,hs_h8_file,dnbr_fn,dnbr_area_extents,fire_after,nofire_before):
		#self.X = []
		#self.Y = []
		#self.W = []
		self.hs_h8_file=hs_h8_file
		self.dnbr_fn=dnbr_fn
		self.dnbr_area_extents=dnbr_area_extents
		self.fire_after=fire_after
		self.nofire_before=nofire_before
		self.hs_h8 = np.loadtxt('dnbr_'+self.hs_h8_file,skiprows=0,delimiter=',')
		self.dnbr_dir = '/g/data/r78/lsd547/landsat_dnbr/' # /local/r78
		self.dnbr_area_fn=self.dnbr_dir+self.dnbr_fn
		self.dnbr_area=np.loadtxt(self.dnbr_area_fn)
		self.dnbr_threshold = 0.27
		self.cw = 40
	def localise(self,x,y):
		(nx,ny) = self.dnbr_area.shape
		#print "DNBR size"
		#print dnbr_area.shape
		#print (x,y)
		w = self.dnbr_area_extents[0]-self.dnbr_area_extents[1]
		h = self.dnbr_area_extents[3]-self.dnbr_area_extents[2]
		tx = nx * (x - self.dnbr_area_extents[1]) / w
		ty = ny * (y - self.dnbr_area_extents[2]) / h
		return (tx,ty)
	def getdnbr(self,x,y):
		(tx,ty) = self.localise(x,y)
		#return np.mean(self.dnbr_area[int(tx-40):int(tx+40),int(ty-40):int(ty+40)])
		return np.nanpercentile(self.dnbr_area[int(ty-self.cw):int(ty+self.cw),int(tx-self.cw):int(tx+self.cw)],95)
		#try:
		#	return np.nanmax(self.dnbr_area[int(ty-self.cw):int(ty+self.cw),int(tx-self.cw):int(tx+self.cw)])
		#except ValueError:
		#	return None
	def isfire(self,x,y,d):
		# use date 2016-01-14 04:12:00 to classify training set.
		# TODO clear hs between midnight and 4:12am
		if d >= self.fire_after and self.getdnbr(x,y) >= self.dnbr_threshold:
			return 1
		else:
			return 0
	def classConfidence(self,x,y,d):
		if d < self.fire_after:
			return 1
		sf = 0.3
		return np.clip(np.absolute(self.dnbr_threshold - self.getdnbr(x,y)),0,sf) / sf # * 4 / 3 #* 0.5
	def hasdnbr(self,x,y):
		if self.dnbr_area_extents[1] + self.cw <= x <= self.dnbr_area_extents[0] - self.cw and self.dnbr_area_extents[3] +self.cw <= y <= self.dnbr_area_extents[2] - self.cw and np.isfinite(self.getdnbr(x,y)).all():
			return 1
		else:
			return 0
	def preprocess(self):
		#trainrows = [hasdnbr(x,y) for x,y in np.nditer(hs_h8[:,[0,1]])]
		self.nanrows = []
		index = 0
		for row in self.hs_h8:
			if not self.hasdnbr(row[0],row[1]) or self.nofire_before < row[2] < self.fire_after:
				self.nanrows.append(index)
			index += 1
		
		#hs_cols = 9
		#hs_cols + 1: dval
		#hs_cols + 2: ndvival
		#hs_cols + 3: rawndvival
		#hs_cols + 4: dnbrval
		#hs_cols + 5: nbrval
		#hs_cols + 6: rawnbrval
		#hs_cols + 7: b4val
		#hs_cols + 8: b6val
		#hs_cols + 9: b7val
		#hs_cols + 10: b14val
		#hs_cols + 11: b4avgval
		#hs_cols + 12: b6avgval
		#hs_cols + 13: b7avgval
		#hs_cols + 14: b14avgval
		#hs_cols + 15: gross_val
		#hs_cols + 16: thin_val
		#hs_cols + 17: fog_val
		#hs_cols + 18: fog2_val
		#hs_cols + 19: daytime flag
		#hs_cols + 20: epochdelta
		#hs_cols + 21: row
		#hs_cols + 22: col
		#hs_cols + 0:  in date range of band data
		#X = hs_h8[:,[12,15]]
		#X = hs_h8[:,[10,11,12,13,14,15,16,17,18,19,20,21,22,24]]
		#X = hs_h8[:,[12,15,16,17,18,19,20,21,22,24]]
		#self.X = self.hs_h8[:,[6,7,8,9,12,15,19,20,21,22,24]]
		#self.X = self.hs_h8[:,[8,9,12,13,15,16,17,18,19,20,21,22,23,24,25,26,27]]
		#self.X = self.hs_h8[:,[8,11,14,16,17,18,20,21,22,23,24,25,26]]
		self.X = self.hs_h8[:,[8,9,11,14,17,21,23,24,25,26]]
		(xrows, xcols) = self.X.shape
		rings = 3
		ringradius = 2000
		ringradius_err = 100
		self.X = np.concatenate((self.X,np.ones((xrows,2*rings + 1))*9999),axis=1)
		#for i in xrange(xrows):
		(lx,ly,le) = (self.hs_h8[:,0],self.hs_h8[:,1],self.hs_h8[:,2])
		# find latest coincident hotspot before this hotspot
		dx = np.ones((xrows,rings)) * 9999
		de = np.ones((xrows,rings)) * 9999
		#for j in xrange(i):
		for j in xrange(1,xrows):
			#ndx = math.sqrt((self.hs_h8[0:j,0]-lx[0:j])**2 + (self.hs_h8[0:j,1]-ly[0:j])**2)
			ndx = np.sqrt(np.square(np.roll(self.hs_h8[:,0],j)-lx) + np.square(np.roll(self.hs_h8[:,1],j)-ly))
			nde = le-np.roll(self.hs_h8[:,2],j)
			#if nde < 0:
			#	print 'you must sort the input hotspot in ascending chronological order'
			#	sys.exit(0)
			for radix in xrange(rings):
				tocopy = np.where( ((radix - 1) * ringradius + ringradius_err < ndx) & (ndx <= radix * ringradius + ringradius_err) & (0 <= nde) & (nde < de[:,radix]),np.ones(xrows),np.zeros(xrows))
				de[:,radix] = np.where(tocopy,nde,de[:,radix])
				dx[:,radix] = np.where(tocopy,ndx,dx[:,radix])
				#if (radix - 1) * ringradius + ringradius_err < ndx <= radix * ringradius + ringradius_err and 0 <= nde < de[radix]:
				#	de[radix] = nde
				#	dx[radix] = ndx
		for radix in xrange(rings):
			self.X[:,xcols + radix + 0] = de[:,radix]
			self.X[:,xcols + radix + 1] = dx[:,radix]
		# Offset time of day. Is this necessary?
		#t = self.hs_h8[:,2] - (10.0/24)
		#self.X[:,xcols+rings*2] = t - int(t)
		# Delete nanrows
		self.X = np.delete(self.X,self.nanrows,axis=0)
		Yrows = self.hs_h8[:,[0,1,2]]
		Yrows = np.delete(Yrows,self.nanrows,axis=0)
		#Y = np.array([isfire(x,y,d) for x,y,d in np.nditer(hs_h8[:,[0,1,2]])])
		self.Y = np.array([self.isfire(row[0],row[1],row[2]) for row in Yrows])
		#self.W = np.absolute((np.clip(np.array([self.getdnbr(row[0],row[1]) for row in Yrows]),0,1) - self.dnbr_threshold) * (2*self.Y-1) * 0.5 + 0.5)
		self.W = np.array([self.classConfidence(row[0],row[1],row[2]) for row in Yrows])

regions = [FireRegion(hs_h8_file='WA_20160101_30_H8_ALBERS_W.csv',
		dnbr_fn='waroona_landsat_dnbr.txt',
		dnbr_area_extents=[-1442387.5,-1530912.5,-3654112.5,-3708912.5],
		fire_after=datestr2num('2016-01-06 08:12:00'),
		nofire_before=datestr2num('2016-01-06 00:12:00')),
	FireRegion(hs_h8_file='WA_20160101_30_H8_ALBERS_E.csv',
		dnbr_fn='waroona_landsat_dnbr.txt',
		dnbr_area_extents=[-1442387.5,-1530912.5,-3654112.5,-3708912.5],
		fire_after=datestr2num('2016-01-06 08:12:00'),
		nofire_before=datestr2num('2016-01-06 00:12:00')),
	FireRegion(hs_h8_file='SA_regional_20151201_31_H8_ALBERS_W.csv',
		dnbr_fn='sa_regional_subset_landsat_dnbr.txt',
		dnbr_area_extents=[717437.5,567912.5,-3235887.5,-3380187.5],
		fire_after= datestr2num('2015-12-02 04:00:00'),
		nofire_before= datestr2num('2015-12-02 00:00:00')),
	FireRegion(hs_h8_file='SA_regional_20151201_31_H8_ALBERS_E.csv',
		dnbr_fn='sa_regional_subset_landsat_dnbr.txt',
		dnbr_area_extents=[717437.5,567912.5,-3235887.5,-3380187.5],
		fire_after= datestr2num('2015-12-02 04:00:00'),
		nofire_before= datestr2num('2015-12-02 00:00:00')),
	FireRegion(hs_h8_file='TAS_Jan01_21_H8_ALBERS_dt_W.csv',
		dnbr_fn='tas_landsat_dnbr.txt',
		dnbr_area_extents=[1159237.5,1074712.5,-4514812.5,-4598387.5],
		fire_after= datestr2num('2016-01-14 04:12:00'),
		nofire_before= datestr2num('2016-01-13 20:12:00')),
	FireRegion(hs_h8_file='TAS_Jan01_21_H8_ALBERS_dt_E.csv',
		dnbr_fn='tas_landsat_dnbr.txt',
		dnbr_area_extents=[1159237.5,1074712.5,-4514812.5,-4598387.5],
		fire_after= datestr2num('2016-01-14 04:12:00'),
		nofire_before= datestr2num('2016-01-13 20:12:00')),
	FireRegion(hs_h8_file='TAS_Jan01_21_H8_ALBERS_dt.csv',
		dnbr_fn='tas_landsat_dnbr.txt',
		dnbr_area_extents=[1159237.5,1074712.5,-4514812.5,-4598387.5],
		fire_after= datestr2num('2016-01-14 04:12:00'),
		nofire_before= datestr2num('2016-01-13 20:12:00')),
	]

for r in regions:
	r.preprocess()

#action = 'pca'
action = 'classify2'
if action=='classify2':
	train_rgn = regions[0] #Waroona W
	train_rgn2 = regions[2] # SA Regional subset W
	train_rgn3 = regions[4] # TAS W
	test_rgn2 = regions[1] #Waroona E
	test_rgn = regions[3] # SA E
	test_rgn3 = regions[5] # TAS E

	# separate into train and test

	traintestsel = 'wholearea'
	#traintestsel = 'randmixed'
	if traintestsel == 'randmixed':
		X = np.concatenate((train_rgn.X,test_rgn.X),axis=0)
		Y = np.concatenate((train_rgn.Y,test_rgn.Y),axis=0)
		W = np.concatenate((train_rgn.W,test_rgn.W),axis=0)
		xrows,xcols = X.shape
		print "X shape "
		print (xrows,xcols)
		train_b = np.random.randint(0,2,size=xrows)
		train_l = np.where(train_b)
		test_l = np.where(1-train_b)
		Xtrain = np.delete(arr=X,obj=train_l,axis=0)
		Ytrain = np.delete(arr=Y,obj= train_l,axis=0)
		Wtrain = np.delete(arr=W,obj= train_l,axis=0)
		Xtest = np.delete(X,test_l,axis=0)
		Ytest = np.delete(Y,test_l,axis=0)
		Wtest = np.delete(W,test_l,axis=0)
	else:
		Xtrain = np.concatenate((train_rgn.X,train_rgn2.X,train_rgn3.X),axis=0)
		Ytrain = np.concatenate((train_rgn.Y,train_rgn2.Y,train_rgn3.Y),axis=0)
		Wtrain = np.concatenate((train_rgn.W,train_rgn2.W,train_rgn3.W),axis=0)
		
		Xtest = np.concatenate((test_rgn.X,test_rgn2.X,test_rgn3.X),axis=0)
		Ytest = np.concatenate((test_rgn.Y,test_rgn2.Y,test_rgn3.Y),axis=0)
		Wtest = np.concatenate((test_rgn.W,test_rgn2.W,test_rgn3.W),axis=0)

	print "Train size " + str(len(Xtrain)) + " " + str(len(Ytrain))

	# check for nans
	(rr,cc) = Xtrain.shape # rows cols
	for c_ in xrange(cc):
		if not np.isfinite(Xtrain[:,c_]).any():
			print "Train Column " + c_ + " contains a non-finite value"

	# check for nans
	(rr,cc) = Xtest.shape # rows cols
	for c_ in xrange(cc):
		if not np.isfinite(Xtest[:,c_]).any():
			print "Test Column " + c_ + " contains a non-finite value"

	#clsfr = linear_model.LogisticRegression(C=1e5)
	#clsfr = BaggingClassifier(linear_model.LogisticRegression(C=1e2),n_estimators=100,bootstrap_features=False)
	#clsfr = BaggingClassifier(SVC(verbose=True,probability=True),n_estimators=100,bootstrap_features=False)
	#clsfr = AdaBoostClassifier(SVC(verbose=True,probability=True),n_estimators=50)
	#clsfr = GradientBoostingClassifier(n_estimators=10000, learning_rate=0.3, max_depth=3, random_state=0)
	#clsfr = DecisionTreeClassifier(random_state=0)
	#clsfr = SVC(kernel='linear',degree=3,verbose=True,probability=True)
	clsfr = SVC(verbose=True,probability=True)
	#clsfr = RandomForestClassifier(verbose=True)
	#clsfr = MLPClassifier(solver='lbfgs',verbose=True)

	clsfr.fit(Xtrain, Ytrain, sample_weight=Wtrain)
	#clsfr.fit(Xtrain, Ytrain)
	print "Classes: " + str(clsfr.classes_)

	print "Score: " + str(clsfr.score(Xtest,Ytest))

	Ypred_class_prob = clsfr.predict_proba(Xtest)
	Ypred_classification = clsfr.predict(Xtest)
	#print Ypred_class_prob
	Ypred_0 = Ypred_class_prob[:,0]
	Ypred_1 = Ypred_class_prob[:,1]


	print 'Number of hotspots classified as fire (Positive class): ' + str(np.sum(Ytest)) + " vs predicted " + str(np.sum(Ypred_classification)) + " or ll weighted " + str(np.sum(Ypred_1))
	print 'Number of hotspots classified as non-fire (Negative class): ' + str(np.sum(1-Ytest)) + " vs predicted " + str(np.sum(1-Ypred_classification)) + " or ll weighted " + str(np.sum(Ypred_0))

	Ypred_0_true = np.delete(Ypred_0,np.where(Ytest),axis=0)
	Ypred_1_true = np.delete(Ypred_1,np.where(1-Ytest),axis=0)
	Ypred_0_false = np.delete(Ypred_0,np.where(1-Ytest),axis=0)
	Ypred_1_false = np.delete(Ypred_1,np.where(Ytest),axis=0)

	print "Predicted lh false negative scores"
	print np.sum(Ypred_0_false)
	print "Predicted lh true negative scores"
	print np.sum(Ypred_0_true)
	print "Predicted lh true positive scores"
	print np.sum(Ypred_1_true)
	print "Predicted lh false positive scores"
	print np.sum(Ypred_1_false)

	print "Predicted false negative scores"
	print np.sum(np.logical_and(1-Ypred_classification,Ytest))
	print "Predicted true negative scores"
	print np.sum(np.logical_and(1-Ypred_classification,1-Ytest))
	print "Predicted true positive scores"
	print np.sum(np.logical_and(Ypred_classification,Ytest))
	print "Predicted false positive scores"
	print np.sum(np.logical_and(Ypred_classification,1-Ytest))

	print "Test size " + str(len(Xtest)) + " " + str(len(Ytest)) + " " + str(len(Ypred_0))

	#f, ((ax1,ax2),(ax3,ax4), (ax5,ax6)) = plt.subplots(3,2)
	f, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)

	ax3.hist(Ypred_1_true,10)
	ax3.set_title('True Positive Rate')
	ax4.hist(Ypred_0_true,10)
	ax4.set_title('True Negative Rate')

	fpr,tpr,thresholds = metrics.roc_curve(Ytest,Ypred_1,pos_label=1,drop_intermediate=False)
	roc_auc = metrics.auc(fpr,tpr)

	#print "FPR"
	#print fpr
	#print "TPR"
	#print tpr
	#print "thresholds"
	#print thresholds

	fprc,tprc,thresholdsc = metrics.roc_curve(Ytest,Ypred_classification,pos_label=1,drop_intermediate=False)
	roc_aucc = metrics.auc(fprc,tprc)

	#print "FPR"
	#print fprc
	#print "TPR"
	#print tprc
	#print "thresholds"
	#print thresholdsc

	lw = 2

	ax1.plot(fpr, tpr, color='darkorange',
	         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
	ax1.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	ax1.set_xlim([0.0, 1.0])
	ax1.set_ylim([0.0, 1.05])
	ax1.set_xlabel('False Positive Rate')
	ax1.set_ylabel('True Positive Rate')
	ax1.set_title('Receiver operating characteristic test data')
	ax1.legend(loc="lower right")

	Ypred2 = clsfr.predict_proba(Xtrain)
	Ypred2_0 = Ypred2[:,0]
	Ypred2_1 = Ypred2[:,1]
	Ypred2_classification = clsfr.predict(Xtrain)
	if True:
		Ypred2_0_true = np.delete(Ypred2_0,np.where(Ytrain),axis=0)
		Ypred2_1_true = np.delete(Ypred2_1,np.where(1-Ytrain),axis=0)
		Ypred2_0_false = np.delete(Ypred2_0,np.where(1-Ytrain),axis=0)
		Ypred2_1_false = np.delete(Ypred2_1,np.where(Ytrain),axis=0)
	
		print 'Number of train hotspots classified as fire (Positive class): ' + str(np.sum(Ytrain)) + " vs predicted " + str(np.sum(Ypred2_classification)) + " or ll weighted " + str(np.sum(Ypred2_1))
		print 'Number of train hotspots classified as non-fire (Negative class): ' + str(np.sum(1-Ytrain)) + " vs predicted " + str(np.sum(1-Ypred2_classification)) + " or ll weighted " + str(np.sum(Ypred2_0))
		print "Predicted Train lh false negative scores"
		print np.sum(Ypred2_0_false)
		print "Predicted Train lh true negative scores"
		print np.sum(Ypred2_0_true)
		print "Predicted Train lh true positive scores"
		print np.sum(Ypred2_1_true)
		print "Predicted Train lh false positive scores"
		print np.sum(Ypred2_1_false)
	
		print "Predicted Train false negative scores"
		print np.sum(np.logical_and(1-Ypred2_classification,Ytrain))
		print "Predicted Train true negative scores"
		print np.sum(np.logical_and(1-Ypred2_classification,1-Ytrain))
		print "Predicted Train true positive scores"
		print np.sum(np.logical_and(Ypred2_classification,Ytrain))
		print "Predicted Train false positive scores"
		print np.sum(np.logical_and(Ypred2_classification,1-Ytrain))
	
		print "Train size " + str(len(Xtrain)) + " " + str(len(Ytrain)) + " " + str(len(Ypred2_0))

	fpr2,tpr2,thresholds2 = metrics.roc_curve(Ytrain,Ypred2_1,pos_label=1)
	roc_auc2 = metrics.auc(fpr2,tpr2)
	
	ax2.plot(fpr2, tpr2, color='darkorange',
	         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc2)
	ax2.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	ax2.set_xlim([0.0, 1.0])
	ax2.set_ylim([0.0, 1.05])
	ax2.set_xlabel('False Positive Rate')
	ax2.set_ylabel('True Positive Rate')
	ax2.set_title('Receiver operating characteristic train data')
	ax2.legend(loc="lower right")
	
	plt.show()
if action=='classify':
	h = .02  # step size in the mesh
	clsfr = linear_model.LogisticRegression(C=1e5)
	
	# we create an instance of Neighbours Classifier and fit the data.
	clsfr.fit(X, Y)
	
	# Plot the decision boundary. For that, we will assign a color to each
	# point in the mesh [x_min, x_max]x[y_min, y_max].
	xaxis = 0
	yaxis = 1
	x_min, x_max = X[:, xaxis].min() - .5, X[:, xaxis].max() + .5
	y_min, y_max = X[:, yaxis].min() - .5, X[:, yaxis].max() + .5
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
	Z = clsfr.predict(np.c_[xx.ravel(), yy.ravel()])
	
	# Put the result into a color plot
	Z = Z.reshape(xx.shape)
	plt.figure(1, figsize=(4, 3))
	plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
	
	# Plot also the training points
	plt.scatter(X[:, xaxis], X[:, yaxis], c=Y, edgecolors='k', cmap=plt.cm.Paired)
	plt.xlabel('NDVI')
	plt.ylabel('B14')
	
	plt.xlim(xx.min(), xx.max())
	plt.ylim(yy.min(), yy.max())
	plt.xticks(())
	plt.yticks(())

	plt.show()
elif action=='pca':
	(xrows,xcols) = X.shape
	print X.shape
	print Y.shape
	#Xscale = scale(np.concatenate((X,np.transpose(Y)),axis=1))
	Xscale = np.concatenate((X,np.zeros((xrows,1))),axis=1)
	Xscale[:,xcols] = Y
	Xscale = scale(Xscale)
	(rws, n_components) = Xscale.shape 
	pca = PCA(n_components=n_components)
	pca.fit(Xscale)
	Xout = pca.transform(Xscale)
	print pca.explained_variance_ratio_
	#PC_grids = {}
	print np.shape(pca.components_)
	#Reshape the principle components into the original grid shape and add them to a dictionary called PCs
	#for i in range(n_components):
	#	PC_grids['PC' + str(i+1)] = pca.components_[i].reshape(shapes)
	#f, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3, sharex='col', sharey='row')
	f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, sharex='col', sharey='row')
	#f, ((ax1, ax2, ax3)) = plt.subplots(1, 3, sharex='col', sharey='row')
	#Xout = Xscale
	ax1.scatter(Xout[:,0],Y, c=Y, edgecolors='k', cmap=plt.cm.Paired)
	#ax1.set_title('KF Diff of NDVI')
	ax1.set_title('RAW NDVI')
	ax2.scatter(Xout[:,1],Y, c=Y, edgecolors='k', cmap=plt.cm.Paired)
	#ax2.set_title('KF NDVI')
	ax2.set_title('Raw NBR')
	ax3.scatter(Xout[:,2],Y, c=Y, edgecolors='k', cmap=plt.cm.Paired)
	#ax3.set_title('RAW NDVI')
	#ax3.set_title('B6')
	ax3.set_title('Gross Cloud')
	ax4.scatter(Xout[:,3],Y, c=Y, edgecolors='k', cmap=plt.cm.Paired)
	#ax4.set_title('KF dNBR')
	#ax4.set_title('B7')
	ax4.set_title('Cirrus Cloud')
	ax5.scatter(Xout[:,4],Y, c=Y, edgecolors='k', cmap=plt.cm.Paired)
	#ax5.set_title('KF NBR')
	#ax5.set_title('B14')
	ax5.set_title('Fog Cloud')
	ax6.scatter(Xout[:,7],Y, c=Y, edgecolors='k', cmap=plt.cm.Paired)
	ax6.set_title('Last coincident hotspot')
	#ax6.set_title('Raw NBR')
	plt.show()

#	hs_h8[i,hs_cols + 1] = dval
#	hs_h8[i,hs_cols + 2] = ndvival
#	hs_h8[i,hs_cols + 3] = rawndvival
#	hs_h8[i,hs_cols + 4] = dnbrval
#	hs_h8[i,hs_cols + 5] = nbrval
#	hs_h8[i,hs_cols + 6] = rawnbrval
#	hs_h8[i,hs_cols + 7] = b6val
#	hs_h8[i,hs_cols + 8] = b7val
#	hs_h8[i,hs_cols + 9] = b14val
#	hs_h8[i,hs_cols + 10] = gross_val
#	hs_h8[i,hs_cols + 11] = thin_val
#	hs_h8[i,hs_cols + 12] = fog_val
#	hs_h8[i,hs_cols + 13] = fog2_val
#	hs_h8[i,hs_cols + 17] = row
#	hs_h8[i,hs_cols + 16] = col
#	hs_h8[i,hs_cols + 15] = epochdelta
#       hs_h8[i,hs_cols + 0] =  in date range of band data

