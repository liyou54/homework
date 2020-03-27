import numpy as np
import matplotlib.pyplot as plt  
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
n = x_data.shape[0]
K = np.dot(x_data,x_data.T)
K = np.power(K,2)
# print(K)

I = np.eye(n)#单位矩阵
One = np.ones((n,n))#全1矩阵
K_center = np.dot(np.dot(I-One/n,K),I-One/n)#根据公式矩阵中心化

# print(K_center)

W = K_center*I
for i in range(n):
    W[i,i]=W[i,i]**(-1/2)
K_normalize = np.dot(np.dot(W,K_center),W)#矩阵归一化
# print(K_normalize)

l = list()
l.append(np.power(x_data,2))
for i in range(1,4):
    for j in range(i):
        l.append((x_data[:,i]*x_data[:,j]).reshape(-1,1)*np.sqrt(2))
res = np.hstack(l)#升维
print(res)
res_centerlize = res - np.mean(res,axis=0)#中心化
res_normalize = res_centerlize/np.linalg.norm(res_centerlize,axis=1).reshape(-1,1)#归一化
print(res_normalize)
print(np.dot(res_normalize,res_normalize.T))
