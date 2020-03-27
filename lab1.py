import numpy as np
import matplotlib.pyplot as plt  
import matplotlib.mlab as mlab
y_label=['g','h']
with open(".\\magic\\magic04.data") as f:
    func = lambda label,data: data.index(label)
    x_data=[]
    y_data=[]
    for line in f:
        data = line.split(",")
        x_data.append(data[:-1])
        y_data.append(func(data[-1][:-1],y_label))
x_data = np.array(x_data,dtype=np.float32)
y_data = np.array(y_data)
# print(x_data)
# print(y_data)

#mean_vector
mean = np.mean(x_data,axis=0)
# print(mean)
#centerlize
center = x_data-mean
# print(center)
#Cov
cov = np.cov(x_data.T)
# print(cov)
#InnerProduct
inn=np.dot(center.T,center)
# print(inn/len(center))
#std
std = np.std(center,axis=0)
print(std)
#OuterProduct
out=0
for i in range(len(center)):
    out = out+center[i].reshape(len(center[0]),1)*center[i]
# print(out/center.shape[0])

#corr(val1,val2)
corr=np.corrcoef(center.T)
print(corr)

plt.scatter(center[:,0],center[:,1])
plt.xlabel('fLength')  #设置X轴标签
plt.ylabel('fWidth') #设置Y轴标签
# plt.show()
plt.savefig('result.png')
plt.close()

a,b = np.histogram(x_data[:,0])
n,bins,patches = plt.hist(x_data[:,0],100,normed=True)
# print(bins)
y=mlab.normpdf(bins,mean[0],std[0])
plt.plot(bins, y)

plt.subplots_adjust()
plt.savefig('result2.png')


print("max: position {},".format(np.argmax(std)),max(std),"\nmin：position {},".format(np.argmin(std)),min(std))
print(np.argmax(cov),np.argmin(cov))
print(cov.reshape(-1)[np.argmax(cov)],cov.reshape(-1)[np.argmin(cov)])