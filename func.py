
import sys
import re
import copy
import math
import numpy

##---- Trims a string towards the left
def TrimLeft( trimString ):
	while( re.match(' ', trimString)):
		trimString = trimString[1:]
	return trimString


##--- concats elements of list and returns a string ---##
def concatList(l1):
	l2 = ''
	if(len(l1)!=0):
		for i in l1:
			l2 = l2+i
	return l2


##--- trims all spaces from a string
def trimStr( str1 ):
	l2 = ''
	for i in str1:
		if(i!=' '):
			l2 = l2+i
	return l2

def trimList( list1 ):
	l2 = []
	for i in list1:
		if(i != '' and i!='\n'):
			l2.append(i)
	return l2


def sgn( value, margin ):
	if( value >= margin):
		return 1
	else:
		return -1

def extendList( l1, maxSize ):
	size = maxSize - len(l1)
	newL = [0]*size
	l1.extend(newL)
	return l1


def permuteDataLabel( xdata, ylabel ):
	newdata = []
	newlabel = []
	shufIdx = numpy.random.permutation(len(xdata))
	if( len(xdata) != len(ylabel)):
		print 'Error......... Mismatch in data and label size..... Exiting'
		sys.exit()

	for i in shufIdx:
		newdata.append(xdata[i])
		newlabel.append(ylabel[i])

	return [newdata, newlabel]
	


def parseInfo( rawData ):
	XData = []
	YData = []
	Fsize = []
	x_temp = []
	seenwt = []

	##---- Do a fast pass of the raw data to get the max feature size
	featureSize = 0
	for flist in rawData:
		for line in flist:
			lineList = line.split()
			size = int(lineList[-1].split(":")[0])
			if(size > featureSize):
				featureSize = size

	for flist in rawData:
		for line in flist:
			lineList = line.split()
			YData.append(int(lineList[0]))
			x_temp = [0]*featureSize
			for i in lineList[1:]:
				arg = i.split(":")
				seenwt.append(int(arg[0]))
				x_temp[int(arg[0])-1] = int(arg[1])
			XData.append(x_temp)
	return [XData, YData, featureSize]

#	for flist in rawData:
#		for line in flist:
#			lineList = line.split()
#			YData.append(int(lineList[0]))
#			currMax = len(x_temp)
#			x_temp = [0]*currMax
#			for i in lineList[1:]:
#				arg = i.split(':')
#				seenwt.append(int(arg[0]))
#				if(int(arg[0])+1 > len(x_temp)):
#					x_temp = extendList(x_temp, int(arg[0])+1)
#				x_temp[int(arg[0])] = int(arg[1])
#			XData.append(x_temp)
#			Fsize.append(len(x_temp))
#	return [XData, YData, Fsize, seenwt]

def parseInfoTest( testData, maxSize ):
	XData  = []
	YData  = []
	Fsize  = []
	x_temp = []

	for flist in testData:
		for line in flist:
			lineList = line.split()
			YData.append(int(lineList[0]))
			x_temp = [0]*maxSize
			for i in lineList[1:]:
				arg = i.split(':')
				if(int(arg[0]) <= maxSize):
					x_temp[int(arg[0])-1] = int(arg[1])
			XData.append(x_temp)
	return [XData, YData]


#def update( wvec, wtxSum, lr, sigmaSq, xvec, ylabel):
#	wvecRet = copy.deepcopy(wvec)
#	x = ylabel*wtxSum
##	if(x < 0):
#	temp = 1/(1 + math.exp(x))
##	else:
##		temp = math.exp(-x)/(1 + math.exp(-x))
#	for j in range(0,len(wvec)):
#		regcomp = (float(2*wvec[j])/sigmaSq)
#		wvecRet[j] = wvec[j] - lr*( -float(ylabel*xvec[j])*temp + regcomp )
#	return wvecRet

def update( wvec, wtxSum, lr, sigmaSq, xvec, ylabel):
	wvecRet = copy.deepcopy(wvec)
	lrfactor = (1 - float(2*lr)/sigmaSq)

	denom = (1 + math.exp(ylabel*wtxSum))
	xvecRet = [ float(lr*ylabel*xvec[j])/denom for j in range(0,len(xvec))]

	wvecRet = [wvec[j]*lrfactor + xvecRet[j] for j in range(0,len(wvec))]

	return wvecRet


def lossFunc(wvec, xvec, ylabel, sigmaSq):
	wtxSum = numpy.dot(wvec,xvec)
	return math.log(1 + math.exp(-ylabel*wtxSum)) + numpy.dot(wvec,wvec)/sigmaSq
	

def evalGrad(xvec, ylabel, wtxSum):
	denom = 1/(1 + math.exp(ylabel*wtxSum))
	grad = [denom*ylabel*xvec[j] for j in range(0,len(xvec))]
	return grad

def updateBatch(wvec, lr, sigmaSq, xvec, ylabel, grad):
	wvecRet = copy.deepcopy(wvec)
	regcomp = [2*wvec[j]/sigmaSq for j in range(0,len(wvec))]
	wvecRet = [wvec[j] - lr*grad[j] -lr*regcomp[j] for j in range(0,len(wvec))]
	return wvecRet
	


#def LogReg(xdata, ydata, wsize, sigmaSq, lr0, epochs, neglog):
#	wvec = []
#	bias = 0
#	mistakeCounter = 0
#	neglogdata = []
#	wvec = [0]+[0]*wsize # adding index for the bias term
#	#wvec = numpy.random.normal(0, 0.1, len(wvec))
#	t = 0
#	lr = 0
#	k = 6414
#	for ep in range(0,epochs):
#		#print "============= Start Epoch ===================="
#		[xdata,ydata] = permuteDataLabel(xdata,ydata)
#		count = 0
#		grad = [0]*len(wvec)
#		for i in range(0,len(xdata)):
#			if(count < k and i!=len(xdata)-1):
#				count = count + 1
#				print count
#				xvec = [1]+xdata[i]
#				ylabel = ydata[i]
#				wtxSum = numpy.dot(wvec,xvec)
#				ExGrad = evalGrad(xvec, ylabel, wtxSum)
#				grad = [grad[j] + ExGrad[j] for j in range(0,len(xvec))]
#			elif(count != 0):
#				grad = [-float(grad[j])/count for j in range(0,len(wvec))]
#				lr = lr0/(1 + (float(lr0*t)/sigmaSq))
#				t = t+1
#				wvec1 = updateBatch(wvec, lr, sigmaSq, xvec, ylabel, grad)
#				print wvec1
#				wvec = wvec1
#				count = 0
#				grad = [0]*len(wvec)
#
#		##---- Make Predictions on the training data ----
#		mistakeCounter = 0.0
#		sumLog = 0.0
#		for i in range(0,len(xdata)):
#			xvec = [1]+xdata[i]
#			wtxSum = numpy.dot(wvec,xvec)
#			#print wtxSum
#			if( wtxSum*ydata[i] < 0 ):
#				mistakeCounter = mistakeCounter + 1.0
#			##--- evluate the negative log likelihood ---
#			if(neglog==1):
#				sumLog = sumLog + (math.log(1 + math.exp(-ylabel*wtxSum)))
#			sumLog = sumLog + numpy.dot(wvec,wvec)/sigmaSq
#		if(neglog==1):
#			neglogdata.append([ep,sumLog])
#			print "Epoch = ",ep,", NegLogLikeliHood = ",sumLog
#		#print "============= End Epoch ===================="
#				
#	if(neglog==0):
#		return [wvec, mistakeCounter, lr]
#	else:
#		return [wvec, mistakeCounter, lr, neglogdata]


def LogReg(xdata, ydata, wsize, sigmaSq, lr0, epochs, neglog):
	wvec = []
	bias = 0
	mistakeCounter = 0
	neglogdata = []
	wvec = [0]+[0]*wsize # adding index for the bias term
	#wvec = numpy.random.normal(0, 0.1, len(wvec))
	t = 0
	lr = 0
	for ep in range(0,epochs):
		#print "============= Start Epoch ===================="
		[xdata,ydata] = permuteDataLabel(xdata,ydata)
		for i in range(0,len(xdata)):
		#for i in range(0,100):
			xvec = [1]+xdata[i]
			ylabel = ydata[i]
			wtxSum = numpy.dot(wvec,xvec)
			#print "wtx:", wtxSum, "yalebl:", ylabel
			lr = lr0/(1 + (float(lr0*t)/sigmaSq))
			t = t+1
			wvec1 = update(wvec, wtxSum, lr, sigmaSq, xvec, ylabel)
			wvec = wvec1

		##---- Make Predictions on the training data ----
		mistakeCounter = 0.0
		sumLog = 0.0
		for i in range(0,len(xdata)):
			xvec = [1]+xdata[i]
			wtxSum = numpy.dot(wvec,xvec)
			#print wtxSum
			if( wtxSum*ydata[i] < 0 ):
				mistakeCounter = mistakeCounter + 1.0
			##--- evluate the negative log likelihood ---
			if(neglog==1):
				sumLog = sumLog + (math.log(1 + math.exp(-ylabel*wtxSum)))
			sumLog = sumLog + numpy.dot(wvec,wvec)/sigmaSq
		if(neglog==1):
			neglogdata.append([ep,sumLog])
			print "Epoch = ",ep,", ObjectiveFunValue = ",sumLog, "L2-Norm-Weight = ",numpy.linalg.norm(wvec)
		#print "============= End Epoch ===================="
				
	if(neglog==0):
		return [wvec, mistakeCounter, lr]
	else:
		return [wvec, mistakeCounter, lr, neglogdata]


def LogRegTest( wvec , xdata, ydata ):
	mistakeCounter = 0.0
	for i in range(0,len(xdata)):
		xvec = [1]+xdata[i]
		wtxSum = numpy.dot(wvec, xvec)
		if( wtxSum*ydata[i] < 0):
			mistakeCounter = mistakeCounter + 1.0
	#print "Test Mistakes = ", mistakeCounter
	return mistakeCounter
	
