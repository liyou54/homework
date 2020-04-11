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
y_data = np.array(y_data)

class Tree():
    #  全局变量 DATA:属性 LABELLIST：标签 RES结果
    DATA = None
    LABELLIST = None
    RES = {1:np.array([],dtype = np.int64),2:np.array([],dtype = np.int64),0:np.array([],dtype = np.int64)}
    # 构造函数 entropy ：熵  index:索引 divide:分类依据 purity：纯度 label:树所属标签
    def __init__(self,index,entropy,divide = ['Nan','Nan'],left=None,right=None,DATA=None,LABELLIST=None):
        self.index = index
        self.count = index.shape[0]
        self.entropy = Tree.Entropy(index)
        self.left = left
        self.right = right
        self.divide = divide
        # print(index)
        if(Tree.DATA is None):
            if(DATA is None):
                raise('请输入DATA')
            Tree.DATA = DATA
        if(Tree.LABELLIST is None):
            if(LABELLIST is None):
                raise('请输入Labellist')
            Tree.LABELLIST = LABELLIST
        dic = Tree.all_np(Tree.LABELLIST[index])
        self.purity = max(dic.values())/sum(dic.values())
        self.label = max(dic,key = dic.get)
        print('类别标签',self.label,'\n纯度为',self.purity)
        print('分类依据：第{}个属性，划分位置{}'.format(self.divide[0],self.divide[1]))
        print('各组分所占比例',dic)
        # 构造树
    def build(self):
        if(self.count<5 or self.purity>0.95):
            Tree.RES[self.label]= np.c_[Tree.RES[self.label].reshape((1,-1)),self.index.reshape((1,-1))]
            return
        else:
            res = self.Divide(Tree.DATA[self.index],Tree.LABELLIST[self.index])
            self.left = Tree(res['index'][0],res['entropy'][0],res['divide'])
            self.right= Tree(res['index'][1],res['entropy'][1],res['divide'])
            self.left.build()
            self.right.build()

    # 统计数组各个数出现次数
    @classmethod
    def all_np(cls,arr):
        key = np.unique(arr)
        result = {}
        for k in key:
            mask = (arr == k)
            arr_new = arr[mask]
            v = arr_new.size
            result[k] = v
        return result
    # 计算熵
    @classmethod
    def Entropy(cls,label,weight=1.):
        count = label.shape[0]
        dic = cls.all_np(label)
        entropy = 0.
        for value in dic.values():
            entropy += value*np.log2(count/value)
        return entropy/(count*weight)
    # @classmethod
    # 分裂成子树
    def Divide(self,data,label):
        n = data.shape[1]
        res = None
        V = 999.
        for i in range(n):
            key = np.sort(np.unique(data[:,i]))
            for j in key[1:-1]:
                index1 = self.index[np.where(data[:,i]>=j)[0]]
                index2 = self.index[np.where(data[:,i]<j)[0]]
                widget1 = (index1.shape[0])/(index1.shape[0]+index2.shape[0])
                widget2 = (index2.shape[0])/(index1.shape[0]+index2.shape[0])
                entorpy1 = Tree.Entropy(Tree.LABELLIST[index1])
                entorpy2 = Tree.Entropy(Tree.LABELLIST[index2])
                if(V>widget1*entorpy1+widget2*entorpy2):
                    V = widget1*entorpy1+widget2*entorpy2
                    res = {'divide':[i,j],'entropy':[entorpy1,entorpy2],'index':[index1,index2]}
        return res



tree = Tree(np.arange(x_data.shape[0]),None,DATA = x_data,LABELLIST = y_data)
tree.build()


for i in range(3):
    print('{}类所含标签：'.format(i),tree.RES[i])
    print('{}类实际情况：'.format(i),y_data[tree.RES[i]])