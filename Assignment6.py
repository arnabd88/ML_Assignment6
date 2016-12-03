

import sys
import re
import copy
import func
import math
import numpy

fold_index = -1
test_index = -1
foldValue = 0


LRateList = [1,0.1,0.01]
SigmaList = [1,0.1,0.01]

trainFileHandle = []
testFileHandle = []

if ('-fold' in sys.argv):
	fold_index = sys.argv.index('-fold')
if ('test' in sys.argv):
	test_index = sys.argv.index('test')

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
	if(test_index != -1):
		[testXData, testYData] = func.parseInfoTest(testFileHandle, FSize)
		print "From Test:", len(testXData), len(testYData)
	sigma = 0.1
	lr = 0.1
	[WVec, LearningMistake] = func.LogReg(XData, YData, FSize, sigma, lr)
	#TestMistakes = func.LogRegTest(



Run_Q1()
