import numpy as np
import matplotlib.pyplot as plt  
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
import numpy as np
from PCA import pca,zeroMean
from LineTree import Tree
x_label = ['萼片长度','萼片宽度','花瓣长度','花瓣宽度']
y_label = ['Iris-virginica', 'Iris-versicolor', 'Iris-setosa']
with open(".//datasets//iris.txt") as f:
    func = lambda label,data: data.index(label)
    x_data=[]
    y_data=[]
    for line in f:
        data = line.split(",")
        x_data.append(data[:-1])
        y_data.append(func(data[-1][1:-2],y_label))
x_data = np.array(x_data,dtype=np.float32)
y_data = np.array(y_data)
print(x_data[:,2])
# x_data2 = pca(x_data,2)

# print(x_data2.shape)
treeList = [
    Tree(np.arange(x_data.shape[0]),[min(x_data[:,i]),max(x_data[:,i])],MINR=0.3,DATA=x_data,asix=i)
    for i in range(x_data.shape[1])
]
# print(1)
# print(x_data[[treeList[0].find([3,5],treeList[0],x_data[:,0])],0])

# def AddGaussian(center,h):
#     #求函数值
#     n = center.shape[0]
#     num = 1/((2*np.pi)**(center.shape[1]/2)*(h**center.shape[1]))
#     print(num)
#     def func(x):
#         res = 0
#         for i in center:
#             res += np.exp(np.dot(i-x,i-x)/(-2*h*h))
#         return num*res/n
#     return func

# def Gradient(center,h):
#     #求梯度
#     n = center.shape[0]
#     num = 1/((2*np.pi)**(center.shape[1]/2)*(h**center.shape[1]))
#     def func(x):
#         res = np.array([0. for i in range(center.shape[1])])
#         for i in center:
#             res+=(i-x)*np.exp(np.dot(i-x,i-x)/(-2*h*h))
#         return res*num/n
#     return func





class DENCLUE():
    def __init__(self,TreeList ,h=None, eps=1e-4, min_density=0.15):
        self.h = h
        self.eps = eps
        self.min_density = min_density
        self.TreeList = TreeList
    def gaussian(self,x,y,h):
        #注意到均值漂移时常数项会被约掉，因而高斯函数可以不计算常数值
        # np.exp(-(np.linalg.norm(x - y) / h) ** 2. / 2.) / ((2. * np.pi) ** (degree / 2)
        return np.exp(-(np.linalg.norm(x - y) / h) ** 2. / 2.)

    def _climbStep(self,point,X,h,W=None):
        n = X.shape[0]
        d = X.shape[1]
        superweight = 0.
        point1 = np.zeros((1, d))
        if W is None:
            W = np.ones((n, 1))
        else:
            W = W
        for j in range(n):
            kernel = self.gaussian(point, X[j], h)
            kernel = kernel * W[j]
            superweight = superweight + kernel
            point1 = point1 + (kernel * X[j])
        point1 = point1 / superweight  # 计算出吸引子
        density = superweight /((h ** d)*np.sum(W))  # 计算出吸引子的密度
        return [point1, density]  # 返回吸引子以及吸引子的密度
    def _climb(self,point, X, h, eps,  W=None):  # 默认每个样本的每个属性权值相等
        error = 99#初始化误差值最大
        prob = 0.
        point1 = np.copy(point)
        radius_new = 0.
        radius_old = 0.
        radius_twiceold = 0.
        # 初始化半径
        iters = 0.
        while True:
            radius_thriceold = radius_twiceold
            radius_twiceold = radius_old
            radius_old = radius_new
            #半径更新
            point0 = np.copy(point1)
            point1, density = self._climbStep(point0, X,  h,W=W)
            error = density - prob#当前容差值
            prob = density
            radius_new = np.linalg.norm(point1 - point0)
            radius = radius_thriceold + radius_twiceold + radius_old + radius_new
            iters += 1
            if iters > 3 and error < eps:
                break
        return [point1, prob, radius]  # 返回吸引子、吸引子的密度、吸引子半径

    def divide(self,data,treeR,h,eps):
        result = dict()
        for i in range(data.shape[0]):
            isrange = [self.TreeList[j].find([data[i,j]-treeR,data[i,j]+treeR],self.TreeList[j],data[:,j]).astype(np.int)
                        for j in range(data.shape[1])]
            X = isrange[0]
            for j in range(1,data.shape[1]):
                X= np.intersect1d(X,isrange[j])
            point,prob,radius = self._climb(data[i],data[X],h,eps)
            if(prob>eps):
                result[i]= [point,prob,radius]
                # print(i,result[i])
        return result
    
    def combine(self,result):
        gra = nx.Graph()
        gra.add_nodes_from(result.keys())
        keys = list(result.keys())
        for i in range(1,len(keys)):
            for j in range(i):
                if(np.linalg.norm(result[keys[i]][0]-result[keys[j]][0])<=1.5*(result[keys[i]][2]+result[keys[j]][2])):
                    gra.add_edge(i,j)

        return gra


den = DENCLUE(treeList)

res = den.divide(x_data,2,0.25,0.0001)
gra = den.combine(res)
print(gra.number_of_nodes())
for c in nx.connected_components(gra):
    print(len(c))
    print(y_data[list(c)])
# f1 = Gradient(np.array([[1,0],[2,1],[3,1],[4,2],[5,1],[2,2]]),1)
# learnRate = 0.05
# stopGradient = 0.05
# h = np.std(x_data2[:,:2]) / 6


# fig = plt.figure()
# ax = Axes3D(fig)
# f = AddGaussian(x_data2,h)
# r = []
# X = np.arange(min(x_data2[:,0]),max(x_data2[:,0]),0.1)
# Y = np.arange(min(x_data2[:,1]),max(x_data2[:,1]),0.1)

# Z = []
# X, Y = np.meshgrid(X, Y)
# print(X.shape, Y.shape)
# for i in range(X.shape[0]):
#     Z.append([])
#     for j in range(X.shape[1]):
#         Z[-1].append(f(np.array([X[i,j],Y[i,j]])))
# Z=np.array(Z)
# print(Z.shape)

# P = []
# for i in range(x_data.shape[0]):
#     P.append(f(x_data2[i]))
# P= np.array(P)
# ax.scatter(x_data2[:,0],x_data2[:,1], P,c='r')
# ax.plot_surface(X,Y,Z, rstride=1, cstride=1, cmap='rainbow')
# plt.show()
# plt.close()
