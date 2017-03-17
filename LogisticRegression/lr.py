#coding:utf-8

import numpy as np

class LogisticRegression():
    def __init__(self):
        self._alpha = None
    
    # sigmoid函数,这里的x可以接受一个向量
    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    def _gradDescent(self, featData, labelData, alpha, maxCycles):
        dataMat = np.mat(featData)
	labelMat = np.mat(labelData).transpose()
	m, n = np.shape(dataMat)
	weigh = np.ones((n, 1))
	for i in range(maxCycles):
	    hx = self._sigmoid(dataMat * weigh)
	    error = labelMat - hx
	    weigh = weigh + alpha * dataMat.transpose() * error
	return weigh
    
    def stocGradDescent(self, featData, labelData, alpha, maxCycles):
        dataMat = np.mat(featData)
	labelMat = np.mat(labelData)
        m, n = np.shape(dataMat)
	weights = np.ones((n, 1))
	for i in range(maxCycles):
	    h = self.sigmoid(dataMat * weights)
	    error = h - labelMat
	    weights = weights + alpha * error * dataMat
	return weights
    
    def stocGradDescent_1(self, featData, labelData, maxCycles):
        dataMat = np.mat(featData)
	labelMat = np.mat(labelData)
        m, n = np.shape(dataMat)
	weights = np.ones((n, 1))
        dataIndex = range(m)
        for i in range(maxCycles):
	    for i in range(m):
	        alpha = 4 / (1.0 + j + i) + 0.01
		randIndex = int(np.random.uniform(0, len(dataIndex)))
		h = self.sigmoid(dataMat[randIndex] * weights)
		weights = weights + alpha * error * dataMat[randIndex]
		del(dataIndex[randIndex])
	return weights
    def fit(self, train_x, train_y, alpha=0.01, maxCycles=100):
        return self._gradDescent(train_x, train_y, alpha, maxCycles)

    def predict(self, test_x, test_y, weigh):
        dataMat = np.mat(test_x)
	labelMat = np.mat(test_y).transpose()
	hx = self._sigmoid(dataMat * weigh)
	m = len(hx)
	error = 0.0
	for i in range(m):
	    if int(hx[i]) > 0.5:
	        print str(i + 1) + '-th sample ', int(labelMat[i]), 'is classfied as: 1'
		if int(labelMat[i]) != 1:
		    error += 1.0
		    print 'classify error.'
	    else:
	        print str(i + 1) + '-th sample ', int(labelMat[i]), 'is classfied as: 0'
		if int(labelMat[i]) != 0:
		    error += 1.0
		    print "classify error."
	error_rate = error/m
	print "error rate is:", "%.4f" % error_rate
	return error_rate
