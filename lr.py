import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn import metrics
from matplotlib.dates import datestr2num, num2date
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import sys
import math

# import some data to play with
#iris = datasets.load_iris()
#X = iris.data[:, :2]  # we only take the first two features.
#Y = iris.target

hs_h8_file='TAS_Jan01_21_H8_ALBERS_dt.csv'
hs_h8 = np.loadtxt('dnbr_'+hs_h8_file,skiprows=0,delimiter=',')

dnbr_area_fn='/local/r78/tas_landsat_dnbr.txt'
dnbr_area=np.loadtxt(dnbr_area_fn)
dnbr_area_extents=[1159237.5,1074712.5,-4514812.5,-4598387.5]
#dnbr_area_extents=[-1442387.5,-1530912.5,-3654112.5,-3708912.5]
def localise(x,y):
	(nx,ny) = dnbr_area.shape
	#print "DNBR size"
	#print dnbr_area.shape
	print (x,y)
	w = dnbr_area_extents[1]-dnbr_area_extents[0]
	h = dnbr_area_extents[3]-dnbr_area_extents[2]
	tx = nx * (x - dnbr_area_extents[1]) / w
	ty = ny * (y - dnbr_area_extents[3]) / h
	return (tx,ty)

def getdnbr(x,y):
	(tx,ty) = localise(x,y)
	return np.mean(dnbr_area[tx-20:tx+20,ty-20:ty+20])
	

# hs_cols = 9
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

# use date 2016-01-14 04:12:00 to classify training set.
# TODO clear hs between midnight and 4:12am
fire_after = datestr2num('2016-01-14 04:12:00')
nofire_before = datestr2num('2016-01-13 20:12:00')
def isfire(x,y,d):
	if d >= fire_after and getdnbr(x,y) >= 0.5:
		return 1
	else:
		return 0

def hasdnbr(x,y):
	if dnbr_area_extents[1] <= x <= dnbr_area_extents[0] and dnbr_area_extents[3] <= y <= dnbr_area_extents[2] and np.isfinite(getdnbr(x,y)).all():
		return 1
	else:
		return 0

#trainrows = [hasdnbr(x,y) for x,y in np.nditer(hs_h8[:,[0,1]])]
nanrows = []
index = 0
for row in hs_h8:
	if not hasdnbr(row[0],row[1]) or nofire_before < row[2] < fire_after:
		nanrows.append(index)
	index += 1

#X = hs_h8[:,[12,15]]
#X = hs_h8[:,[10,11,12,13,14,15,16,17,18,19,20,21,22,24]]
#X = hs_h8[:,[12,15,16,17,18,19,20,21,22,24]]
X = hs_h8[:,[8,9,12,15,19,20,21,22,24]]
(xrows, xcols) = X.shape
rings = 5
ringradius = 2000
ringradius_err = 100
X = np.concatenate((X,np.ones((xrows,2*rings + 1))*9999),axis=1)
for i in xrange(xrows):
	for radix in xrange(rings):
		(lx,ly,le) = (hs_h8[i,0],hs_h8[i,1],hs_h8[i,2])
		# find latest coincident hotspot before this hotspot
		(dx,de) = (9999,9999)
		for j in xrange(i):
			ndx = math.sqrt((hs_h8[j,0]-lx)**2 + (hs_h8[j,1]-ly)**2)
			nde = le-hs_h8[j,2]
			if nde < 0:
				print 'you must sort the input hotspot in ascending chronological order'
				sys.exit(0)
			if (radix - 1) * ringradius + ringradius_err < ndx < radix * ringradius + ringradius_err and nde < de:
				de = nde
				dx = ndx
		X[i,xcols + radix + 0] = de
		X[i,xcols + radix + 1] = dx
	t = hs_h8[i,2] - (10.0/24)
	X[i,xcols+rings*2] = t - int(t)
X = np.delete(X,nanrows,axis=0)
Yrows = hs_h8[:,[0,1,2]]
Yrows = np.delete(Yrows,nanrows,axis=0)
#Y = np.array([isfire(x,y,d) for x,y,d in np.nditer(hs_h8[:,[0,1,2]])])
Y = np.array([isfire(row[0],row[1],row[2]) for row in Yrows])
W = np.absolute(np.clip(np.array([getdnbr(row[0],row[1]) for row in Yrows]),0,1) - 1 + Y)

#action = 'pca'
action = 'classify2'
if action=='classify2':
	# separate into train and test
	xrows,xcols = X.shape
	print "X shape "
	print (xrows,xcols)
	train_b = np.random.randint(0,2,size=xrows)
	#print "Train set len " +  str(len(train)) + " +ve = " + str(np.sum(train)) + " zero = " + str(len(train) - np.sum(train))
	train_l = np.where(train_b)
	test_l = np.where(1-train_b)
	# TODO FIXME make this random choice more pythonic e.g. using np.choice()
	#for idx in xrange(len(train_b)):
	#	if train_b[idx]==1:
	#		train_l.append(idx)
	#	else:
	#		test_l.append(idx)
	
	Xtrain = np.delete(arr=X,obj=train_l,axis=0)
	Ytrain = np.delete(arr=Y,obj= train_l,axis=0)
	Wtrain = np.delete(arr=W,obj= train_l,axis=0)
	Xtest = np.delete(X,test_l,axis=0)
	Ytest = np.delete(Y,test_l,axis=0)
	Wtest = np.delete(W,test_l,axis=0)
	print "Train size " + str(len(Xtrain)) + " " + str(len(Ytrain))

	logreg = linear_model.LogisticRegression(C=1e5)
	#logreg = DecisionTreeClassifier(random_state=0)
	#logreg = SVC(kernel='linear',degree=3,verbose=True,probability=True)
	#logreg = RandomForestClassifier(verbose=True)
	logreg.fit(Xtrain, Ytrain, sample_weight=Wtrain)
	Ypred = logreg.predict_proba(Xtest)[:,1]

	Ypred_0 = np.delete(Ypred,np.where(Ytest),axis=0)
	Ypred_1 = np.delete(Ypred,np.where(1-Ytest),axis=0)

	print "Predicted false negative scores"
	print np.sum(Ypred_0)
	print "Predicted true negative scores"
	print len(Ypred_0) - np.sum(Ypred_0)
	print "Predicted true positive scores"
	print np.sum(Ypred_1)
	print "Predicted false positive scores"
	print len(Ypred_1) - np.sum(Ypred_1)

	print "Test size " + str(len(Xtest)) + " " + str(len(Ytest)) + " " + str(len(Ypred))
	#for yp in Ypred:
	#	print yp

	f, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)

	ax3.scatter(np.arange(len(Ytest)),Ytest,c='navy',label='Test')
	ax3.scatter(np.arange(len(Ypred)),Ypred,c='darkorange',label='Predicted')
	ax3.set_title('Classification')
	ax3.legend(loc='lower right')

	fpr,tpr,thresholds = metrics.roc_curve(Ytest,Ypred,pos_label=1,drop_intermediate=False)
	roc_auc = metrics.auc(fpr,tpr)

	print "FPR"
	print fpr
	print "TPR"
	print tpr
	print "thresholds"
	print thresholds

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

	Ypred2 = logreg.predict_proba(Xtrain)[:,1]
	fpr2,tpr2,thresholds2 = metrics.roc_curve(Ytrain,Ypred2,pos_label=1)
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
	logreg = linear_model.LogisticRegression(C=1e5)
	
	# we create an instance of Neighbours Classifier and fit the data.
	logreg.fit(X, Y)
	
	# Plot the decision boundary. For that, we will assign a color to each
	# point in the mesh [x_min, x_max]x[y_min, y_max].
	xaxis = 0
	yaxis = 1
	x_min, x_max = X[:, xaxis].min() - .5, X[:, xaxis].max() + .5
	y_min, y_max = X[:, yaxis].min() - .5, X[:, yaxis].max() + .5
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
	Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])
	
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
	(rws, n_components) = Xscale.shape #index - len(nanrows)
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

