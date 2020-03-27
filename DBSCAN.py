import matplotlib.pyplot as plt
import sklearn
import numpy as np

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
class DBSCAN():
    def __init__(self,r,count):
        self.count = count
        self.r = r

    
    def PickPoint(self,x_data,distance):
        length = len(x_data)
        dis = distance(x_data)
        res = np.array([np.where(dis[i]<self.r,1,0) for i in range(len(dis))])
        m = np.array([-1 for i in range(length)])
        for i in range(length):
            mask = (res[i]==1)
            new = res[i][mask]
            if new.size > self.count:
                m[i]=1
        for i in range(length):
            if(m[i]==-1):
                for j in range(length):
                    if(m[j]==1 and res[i,j]==1):
                        m[i]=0
                        break
        print(dis,res,m)
        return res,m

    def Cluster(self,res,m):
        length = len(m)
        kind = []
        for i in range(length):
            if(m[i]==1):
                queue = []
                kind.append([])
                queue.append(i)
                while(queue != []):
                    tmp = queue.pop(0)
                    kind[-1].append(tmp)
                    m[tmp] = -1
                    for j in range(length):
                        #可达
                        if(res[tmp,j]==1 and m[j]!=-1):
                            if(m[j]==0):
                                kind[-1].append(j)
                                m[j]=-1
                            if(m[j]==1):
                                queue.append(j)
                                m[j]=-1
        return kind

def Euclidean(data):
    x= np.array(
        [
            [np.linalg.norm(data[i] - data[j]) for i in range(len(data))]
            for j in range(len(data))
        ]
    )
    return x
s = DBSCAN(.52,13)



res , m = s.PickPoint(x_data,Euclidean)
print(m)
fenlei = s.Cluster(res,m)
print(fenlei ,[len(i) for i in fenlei])
for i in range(len(fenlei)):
    result = [0,0,0]
    for j in range(len(fenlei[i])):
        result[y_data[fenlei[i][j]]]+=1
    print(result)
