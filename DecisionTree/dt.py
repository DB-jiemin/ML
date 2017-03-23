#coding:utf-8

import numpy as np
import os

class DecisionTree():

    def __init__(self, criteria="ID3"):
        self._tree = None
        if criteria == "ID3" or criteria == "C4.5":
	    self._criteria = criteria
	else:
	    raise Exception("criterion should be ID3 or C4.5")

    def _calEntropy(self, y):
        n = y.shape[0]
	labelCounts = {}
	for label in y:
	    if label not in labelCounts.keys():
	        labelCounts[label] = 1
	    else:
	        labelCounts[label] += 1
	entropy = 0.0
	for key in labelCounts:
	    prob = float(labelCounts[key])/n
	    entropy -= prob * np.log2(prob)
	return entropy

    def _splitData(self, X, y, axis, cutoff):
        ret = []
	featVec = X[:,axis]
	n = X.shape[1]
	X = X[:,[i for i in range(n) if i!=axis]]
	for i in range(len(featVec)):
	    if featVec[i] == cutoff:
	        ret.append(i)
        return X[ret, :], y[ret]
    
    def _chooseBestSplit(self, X, y):
        numFeat = X.shape[1]
	baseEntropy = self._calEntropy(y)
	bestSplit = 0.0
	best_idx = 1
	for i in range(numFeat):
	    featlist = X[:, i]
	    uniqueVals = set(featlist)
	    curEntropy = 0.0
	    splitInfo = 0.0
	    for value in uniqueVals:
	        sub_x, sub_y = self._splitData(X, y, i, value)
		prob = len(sub_y) / float(len(y))
		curEntropy += prob * self._calEntropy(sub_y)
		splitInfo -= prob * np.log2(prob)
	    IG = baseEntropy - curEntropy
	    if self._criteria == "ID3":
	        if IG > bestSplit:
		    bestSplit = IG
		    best_idx = i
	    if self._criteria == "C4.5":
	        if splitInfo == 0.0:
		    pass
		IGR = IG / splitInfo
		if IGR > bestSplit:
		    bestSplit = IGR
		    best_idx = i
        return best_idx

    def _majorityCnt(self, labellist):
        labelCount = {}
	for vote in labellist:
	    if vote in labellist:
	        labelCount[vote] = 0
	    labelCount[vote] += 1
	sortedClassCount = sorted(labelCount.iteritems(), key=lambda x:x[1], reverse=True)
	return sortedClassCount[0][0]

    def _createTree(self, X, y, featureIndex):
        labelList = list(y)
	if labelList.count(labelList[0]) == len(labelList):
	    return labelList[0]
	if len(featureIndex) == 0:
	    return self._majorityCnt(labelList)
	bestFeatIndex = self._chooseBestSplit(X, y)
	bestFeatAxis = featureIndex[bestFeatIndex]
	featureIndex = list(featureIndex)
	featureIndex.remove(bestFeatAxis)
	featureIndex = tuple(featureIndex)
	myTree = {bestFeatAxis:{}}
	featValues = X[:, bestFeatIndex]
	uniqueVals = set(featValues)
	for value in uniqueVals:
	    sub_X, sub_y = self._splitData(X, y, bestFeatIndex, value)
	    myTree[bestFeatAxis][value] = self._createTree(sub_X, sub_y, featureIndex)
	return myTree

    def fit(self, X, y):
        if isinstance(X, np.ndarray) and isinstance(y, np.ndarray):
	    pass
	else:
	    try:
	        X = np.array(X)
		y = np.array(y)
	    except:
	        raise TypeError("numpy.ndarray required for X,y")
        featureIndex = tuple(["x" + str(i) for i in range(X.shape[1])])
	self._tree = self._createTree(X, y,featureIndex)
	return self

    def _classify(self, tree, sample):
        featIndex = tree.keys()[0]
	secondDict = tree[featIndex]
	axis = featIndex[1:]
	key = sample[int(axis)]
	valueOfKey = secondDict[key]
	if type(valueOfKey).__name__ = "dict":
	    return self._classify(valueOfKey, sample)
	else:
	    return valueOfKey
    
    def predict(self, X):
        if self._tree == None:
	    raise NotImplementedError("Estimator not fitted, call fit function first")
	if isinstance(X, np.ndarray):
	    pass
	else:
	    try:
	        X = np.array(X)
            except:
	        raise TypeError("numpy.ndarray required for X")
	if len(X.shape) == 1:
	    return self._classify(self._tree, X)
	else:
	    result = []
	    for i in range(X.shape[0]):
	        value = self._classify(self._tree, X[i])
	        print str(i+1) + "-th sample is classfied as:", value
		result.append(value)
	    return np.array(result)

    def show(self, outpdf):
        if self._tree == None:
	    pass
	import treePlotter
	treePlotter.createPlot(self._tree, outpdf)

if __name__ == "__main__":
    trainfile = r"data/train.txt"
    testfile = r"data/test.txt"
    import sys
    sys.path.append(r"")
    import dataload as dload
    train_x, train_y = dload.loadData(trainfile)
    test_x, train_y = dload.loadData(testfile)
    clf = DecisionTree(criteria="C4.5")
    clf.fit(train_x, train_y)
    result = clf.predict(test_x)
    outpdf = r"tree.pdf"
    clf.show(outpdf)


