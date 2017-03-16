#coding:utf-8
#
# 加法模型
# 把每个基础分类器按照一定的权重加起来
#
import numpy as np

class AdaboostClassifier(object):
    # 初始化
    def __init__(self, max_iter=100, numSteps=10, num_iter=50):
        self._max_iter = max_iter
	self._numSteps = numSteps
	self._num_iter = num_iter
    
    #决策树桩分类器,阈值threshVal很重要,只要保证正确率大于0.5了,才有意义,否则与随机猜测没有区别
    def _stumpClassify(self, X, dimen, threshVal, threshIneq):
        # function:决策树桩,通过阈值比较对数据集进行分类
        # X:数据集
	# dimen:维数,也就是第几列,在第几列上进行比较
	# threshVal:决策阈值
	# threshIneq:小于等于 还是 大于判定为-1
        retArray = np.ones((np.shape(X)[0], 1)) #创建一个(m,1)数组,用来存放决策树桩预测的结果
	if threshIneq == "lt": #小于等于
	    # 语法:X的第dimen列的所有值和threshVal比较,为真的将被赋值-1.0,否则不变
	    retArray[X[:, dimen] <= threshVal] = -1.0 #小于等于 threshVal的为-1
	else:
	    retArray[X[:, dimen] > threshVal] = -1.0
	return retArray

    # 建立决策树桩
    def _buildStump(self, X, y, D):
        # 输入: X, y和样本权重值D
	# 功能:找到最佳的决策树桩
	# 返回:dict(dim, ineq, thresh)错误率 基于该特征和阈值下的类别估计值
	dataMmat = np.mat(X) # 将输入的数据转换成numpy格式
	labelMat = np.mat(y).T # .T 操作表示转置
	m, n = np.shape(dataMat)
	bestStump = {} #存放最佳的决策树桩
	bestClasEst = np.mat(np.zeros((m, 1))) # 用来存放预测出来的结果
	minError = np.inf # 无穷大
	for i in range(n): # 决策树桩是建立在列上面的,所以循环n即可
	    minVal = dataMat[:, i].min() # 获取该列的最小值
	    maxVal = dataMat[:, i].max() # 获取该列的最大值
	    stepSize = (maxVal - minVal)/self._numSteps # 
	    for j in range(-1, int(self._numSteps) + 1): #决策阈值
	        for inequal in ['lt', 'gt']: #用每个阈值判断左右两面的err,然后选取err较小的一个
		    threshVal = (minVal + float(j) * stepSize) # threshVal决策阈值
		    predVals = self._stumpClassify(dataMat, i, threshVal, inequal) # 产生决策树桩
		    ######################################
		    # 下面这两个用到一个小技巧,先假设全正确的
		    # 然后将正确的全部赋值为0,在求和就是错误率err
		    errArr = np.mat(np.ones((m, 1)))
		    errArr[predVals == labelMat] = 0
		    ######################################
		    weightedError = D.T * errArr # 计算所有样本错误率,整体错误率
		    if weightedError < minError: # 和最小错误率进行比较
		        minError = weightedError # 替换最小误差
			bestClasEst = predVals.copy() # 
			bestStump['dim'] = i # 存储第i维,表示在第i维上做的比较
			bestStump['thresh'] = threshVal # 阈值
			bestStump['ineq'] = inequal # 如何比较的,是lt还是gt
	return  bestStump, minError, bestClasEst

    # 拟合数据
    def fit(self, X, y):
        # 输入:X, y
	# 输出:弱分类器weakClassArr(字典)
	weakClassArr = [] # 若分类器列表
	m = np.shape(X)[0] # 获取样本个数
	D = np.mat(np.ones((m, 1))/m) # 初始化每个样本的权重值,默认每个样本的权重都相等
	aggClassEst = np.mat(np.zeros((m, 1))) # 
	for i in range(self._num_iter): # 迭代
	    bestStump, error, classEst = self._buildStump(X, y, D) # 寻找最佳的决策树桩
	    alpha = float(0.5 * np.log(1.0 - error)/max(error,1e-16)) # 每个决策树桩的权重
	    bestStump['alpha'] = alpha # 存放决策树桩的权重
	    weakClassArr.append(bestStump) # 将若分类器存放到weakClassArr
	    ###################################################
	    # 下面两行套公式
	    expon = np.multiply(-1 * alpha * np.mat(y).T, classEst) # -1 * alpha * G(xi) G(*)表示若分类器
	    D = np.multiply(D, np.exp(expon)) # weights * exp(-1 * alpha * G(xi))
	    ###################################################
	    D = D/D.sum() # 归一化
	    aggClassEst += alpha * classEst
	    aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(y).T, np.ones((m, 1))) # 计算预测值和真实label的误差
	    errorRate = aggErrors.sum()/m # 计算总的误差
	    if errorRate == 0.0: # 如果总误差为0.0,退出循环
	        break
	return weakClassEst # 返回所有的若分类器,即AdaBoost模型

    def predict(self, test_X, classifyArr): 
        # 输入:test_X测试集
	#      classifyArr分类器
	# 输出:分类结果(二分类)
	dataMat = np.mat(test_X)
	m = np.shape(dataMat)[0]
	aggClassEst = np.mat(np.zeros((m, 1)))
	for i in range(len(classifierArr)):
	    classEst = self._stumpClassify(dataMat, classifierArr[i]['dim'], classifierArr[i]['thresh'], classifierArr[i]['ineq'])
	    aggClassEst += classifierArr[i]['alpha'] * classEst
	return np.sign(aggClassEst)
