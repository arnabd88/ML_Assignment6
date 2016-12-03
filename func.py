
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
				if(int(arg[0]) < maxSize):
					x_temp[int(arg[0])] = int(arg[1])
			XData.append(x_temp)
	return [XData, YData]



#def Logreg(xdata, ydata, wsize, sigma, lr):
#	wvec = []
#	bias = 0
#	mistakeCounter = 0
#	wvec = [0]+[0]*wsize
#	
#	looprange = len(wvec)
#	for i in range(0,len(xdata)):
#		xvec = [1] + xdata[i]
#		ylabel = ydata[i]
#		wtxSum = 0
#		if(len(xvec) < len(wvec)):
#			looprange = len(xvec)
#		for j in range(0,looprange):
#			wtxSum = wtxSum + wvec[j]*xvec[j]
#		for j in range(0, looprange):
			
			


