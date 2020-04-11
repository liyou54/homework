import numpy as np
class Tree(object):
    MINR = None#线段树最小距离
    DATA = None#线段树数据
    # 线段树初始化
    def __init__(self,index,r,left=None,right=None,MINR=None,DATA = None,asix = None):
        self.left = left#线段树左子树
        self.right = right#线段树右子树
        self.index = index#索引数组
        self.r = r #线段树左右端点
        self.mid = (sum(r)/2)# 线段树中点
        if(Tree.MINR is None):
            if(MINR is None):
                raise('请输入区间范围')
            Tree.MINR = MINR
        if(Tree.DATA is None):
            if(DATA is None):
                raise('请输入数据')
            Tree.DATA = DATA
        if(asix is not None):
            self._build(self,DATA[:,asix])
            print(asix)


    def _build(self,tree,data):
        # 线段树建立，递归调用
        if(len(tree.index)==1 or tree.r[1]-tree.r[0]<=Tree.MINR):
            return
        for i in tree.index:
            if(data[i]<tree.mid):
                if(tree.left is None):
                    tree.left=Tree([],[tree.r[0],tree.mid])
                tree.left.index.append(i)
            else:
                if(tree.right is None):
                    tree.right = Tree([],[tree.mid,tree.r[1]])
                tree.right.index.append(i)
        # print([Tree.DATA[i,1] for i in tree.index])
        if(tree.left):
            tree._build(tree.left,data)
        if(tree.right):
            tree._build(tree.right,data)
    
    def find(self,r,tree,data):
        # 线段树查询递归调用
        if(tree.left is None and tree.right is None):
            res = []
            for i in tree.index:
                if(data[i]>=r[0] and data[i]<=r[1]):
                    res.append(i)
            return np.array(res)
        elif(r[1]>=tree.r[1] and r[0]<=tree.r[0]):
            return tree.index
        res = np.array([],dtype=np.int16)
        if(r[0]>=tree.mid and tree.right is not None):
            res = np.hstack((res,tree.find(r,tree.right,data)))
        elif(r[1]<=tree.mid and tree.left is not None):
            res = np.hstack((tree.find(r,tree.left,data),res))
        elif(r[0]<=tree.mid and r[1]>=tree.mid):
            if(tree.left is not None):
                res = np.hstack((res,tree.find([r[0],tree.mid],tree.left,data)))
            if(tree.right is not None):
                res = np.hstack((res,tree.find([tree.mid,r[1]],tree.right,data)))
        return res
        

