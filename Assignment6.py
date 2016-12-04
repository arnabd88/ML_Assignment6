

import sys
import re
import copy
import func
import math
import numpy

fold_index = -1
test_index = -1
foldValue = 0


LRateList = [0.0001,0.000001]
SigmaSqList = [1,0.1,0.01,0.001]
epochList = [4,5,6]

trainFileHandle = []
testFileHandle = []

if ('-fold' in sys.argv):
	fold_index = sys.argv.index('-fold')
if ('-test' in sys.argv):
	test_index = sys.argv.index('-test')

if(fold_index != -1):
	foldValue = int(sys.argv[fold_index + 1])
	trainFileHandle.append(open(sys.argv[fold_index+2],'r+').read().splitlines())
else:
	print "Training files not provided ........... Exiting!!"
	sys.exit()


if (test_index != -1):
	testFileHandle.append(open(sys.argv[test_index+1],'r+').read().splitlines())
else:
	print 'Test Data Not Found.... No Testing!!'




def Run_Q1():
	
	[XData, YData, FSize] = func.parseInfo(trainFileHandle)
	print "YData: ", len(YData), len(XData), FSize
	sigmaSq = 0.0001
	lr = 1e-4
	[WVec, LearningMistake] = func.LogReg(XData, YData, FSize, sigmaSq, lr, 3)
	if(test_index != -1):
		[testXData, testYData] = func.parseInfoTest(testFileHandle, FSize)
		print "From Test:", len(testXData), len(testYData)
		TestMistakes = func.LogRegTest(WVec, testXData, testYData)
		#print WVec


def Run_kvalidate():
	[XData, YData, FSize] = func.parseInfo(trainFileHandle)
	bestSigmaSq = 0
	MaxAccuracy = 0
	epochs = 1
	lr = 0.00001

	kfoldData = []
	for sigmaSq in SigmaSqList:
		blockSize = len(XData)/foldValue
		testAcc = 0
		trainAcc = 0
		for k in range(0,foldValue):
			print "SigmaSq = ",sigmaSq, ", LR = ",lr
			KXTest = XData[k*blockSize:(k+1)*blockSize]
			KYTest = YData[k*blockSize: (k+1)*blockSize]
			KXData = [XData[i] for i in range(0,len(XData)) if(i<k*blockSize or i>=(k+1)*blockSize)]
			KYData = [YData[i] for i in range(0,len(XData)) if(i<k*blockSize or i>=(k+1)*blockSize)]
			[Wvec, trainMist] = func.LogReg(KXData, KYData, FSize, sigmaSq, lr, epochs)
			testMist = func.LogRegTest(Wvec, KXTest, KYTest)
			trainAcc = trainAcc + 100*float((len(KXData) - trainMist))/len(KXData)
			testAcc  = testAcc + 100*float((len(KXTest) - testMist))/len(KXTest)
		avgtrainacc = trainAcc/foldValue
		avgtestacc  = testAcc/foldValue
		kfoldData.append([sigmaSq, avgtrainacc, avgtestacc])
		if(MaxAccuracy < avgtestacc):
			MaxAccuracy = avgtestacc
			bestSigma = sigmaSq

	print "Best SigmaSq = ",bestSigmaSq, ", using lr = ", lr

	##----------- Now learn on the entire data set --------------------##
	[Wvec, trainMist] = func.LogReg(XData, YData, FSize, bestSigma, lr, 1)
	print "Train Data size = ", len(XData)
	FinalTrainAcc = 100*float(len(XData) - trainMist)/len(XData)
	print "Final Training Accuracy = ", FinalTrainAcc," %"
	if(test_index != -1):
		[testXData, testYData] = func.parseInfoTest(testFileHandle, FSize)
		TestMistakes = func.LogRegTest(Wvec, testXData, testYData)
		print "Test Data size = ", len(testXData)
		FinalTestAcc = 100*float(len(testXData) - TestMistakes)/len(testXData)
		print "Final Test Accuracy = ", FinalTestAcc," %"
	
		

					


#Run_Q1()
Run_kvalidate()
