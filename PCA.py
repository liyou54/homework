import numpy as np
def zeroMean(dataMat):
    # 求各列特征的平均值
    meanVal = np.mean(dataMat, axis=0)
    newData = dataMat - meanVal
    return newData, meanVal
def pca(dataMat,k):  
    newData,meanVal=zeroMean(dataMat)  
    covMat=np.cov(newData,rowvar=0)    #求协方差矩阵

    eigVals,eigVects=np.linalg.eig(np.array(covMat))#求特征值和特征向量,特征向量是按列放的，即一列代表一个特征向量  
    eigValIndice=np.argsort(eigVals)            #对特征值从小到大排序  
    k_eigValIndice=eigValIndice[-1:-(k+1):-1]   #最大的k个特征值的下标  
    k_eigVect=eigVects[:,k_eigValIndice]        #最大的k个特征值对应的特征向量  
    lowDDataMat=np.dot(newData,k_eigVect)               #低维特征空间的数据
    return lowDDataMat
