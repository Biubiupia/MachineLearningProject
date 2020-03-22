import pandas as pd
from sklearn import linear_model
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from scipy import stats
import os

#归一化
def MaxMinNormalization(x,Max,Min):
    x = (x - Min) / (Max - Min);
    return x

def drawPicture(filepath):
    data = pd.read_csv(file_path)
    x = data.iloc[:, :1]
    y = data.iloc[:, 1:]
    # 绘制曲线
    plt.plot(x, y, 'b', label="abs")
    plt.xlabel("WL(nm)")
    plt.ylabel('abs')
    plt.title("all_sample")
    plt.savefig("curve.jpg")


arrFilename = []
arrVar = []
arrStd = []
arrCv = []
arrwaveRate = []
arrSkew = []
arrKurtosis = []
arrCategory = []
arrMaxWL = []
arrPVR = []
ROOT = "E:\datas\chem"
for root, dirs, files in os.walk(ROOT):
    for file_name in files:
        #print(root)
        file_path=os.path.join(root,file_name)
        #print(file_path)
        data = pd.read_csv(file_path)
        x = data.iloc[:, :1]
        # print(x)
        y = data.iloc[:, 1:]
        #绘制曲线
        plt.plot(x, y, 'b', label="abs")

        # 计算那几个特征
        abs = MaxMinNormalization(y, y.max(), y.min())
        # print(abs)

        var = np.var(abs)
        #print("方差为：%f" % var)
        std = np.std(abs, ddof=1)
        #print("标准差为：%f" % std)
        cv = std / np.mean(abs)
        #print("变异系数为：%f" % cv)
        # 计算对数收益率
        # logreturns = diff(log(abs.values.reshape(-1)+0.01))
        # print("变异系数为：%f" % logreturns)
        logreturns = diff(log(abs.values.reshape(-1) + 0.01))
        # print(logreturns)
        waveRate = np.std(logreturns, ddof=1) / np.mean(logreturns)
        waveRate = waveRate / sqrt(1 / abs.shape[0])
        #print("波动率为：", waveRate)
        # 偏度代表性不强
        skew = stats.skew(abs)
        #print("偏度为：%f" % skew)
        kurtosis = stats.kurtosis(abs)
        #print("峰度为：%f" % kurtosis)
        maxIndex = abs.idxmax(axis=0)
        maxWL = np.array(x)[maxIndex]
        #print("文件名：%s" % file_name,"最大值索引为：%d" % maxIndex,"最大值所在波长:%d" % maxWL)
        #要用归一化前的数据计算峰谷比
        peak = y.max()
        valley = y.min()+0.01
        PVR = peak/valley
        #print("峰谷比为：%f" % PVR)
        
        #加入数组中
        arrFilename.append(file_name)
        arrVar.append(var)
        arrStd.append(std)
        arrCv.append(cv)
        arrwaveRate.append(waveRate)
        arrSkew.append(skew)
        arrKurtosis.append(kurtosis)
        arrMaxWL.append(maxWL)
        arrPVR.append(PVR)
        if root == os.path.join(ROOT,"cluster"):
            arrCategory.append("cluster")
        elif root == os.path.join(ROOT,"complex"):
            arrCategory.append("complex")
        elif root == os.path.join(ROOT,"partical"):
            arrCategory.append("partical")
        elif root == os.path.join(ROOT,"poly-disperse"):
            arrCategory.append("poly-disperse")
        else:
            arrCategory.append("unknown")
            
        
arrFilename = np.array(arrFilename).reshape(-1)
arrVar = np.array(arrVar).reshape(-1)
arrStd = np.array(arrStd).reshape(-1)
arrCv = np.array(arrCv).reshape(-1)
arrwaveRate = np.array(arrwaveRate).reshape(-1)
arrSkew = np.array(arrSkew).reshape(-1)
arrKurtosis = np.array(arrKurtosis).reshape(-1)
arrMaxWL = np.array(arrMaxWL).reshape(-1)
arrPVR = np.array(arrPVR).reshape(-1)
arrCategory = np.array(arrCategory).reshape(-1)
#arrCategory = np.array([0]*arrVar.shape[0])
rst=pd.DataFrame({'filename':arrFilename,'var':arrVar,'std':arrStd,'cv':arrCv,'waveRate':arrwaveRate,'skew':arrSkew,                  'kurtosis':arrKurtosis,'peakPos':arrMaxWL,'PVR':arrPVR,'category':arrCategory})
rst.to_csv("./output2.28.csv",index=False)    #保存在当前文件夹
#显示图像
#plt.legend(loc="upper right")  # 显示图中的标签
plt.xlabel("WL(nm)")
plt.ylabel('abs')
plt.title("all_sample")
plt.show()
rst


# In[2]:


# 导入画图工具
import matplotlib.pyplot as plt
# 导入数组工具
import numpy as np
# 导入数据集生成器
from sklearn.datasets import make_blobs
# 导入KNN 分类器
from sklearn.neighbors import KNeighborsClassifier
# 导入数据集拆分工具
from sklearn.model_selection import train_test_split
 
# 生成样本数为500，分类数为5的数据集，特征数为2
data=make_blobs(n_samples=500, n_features=2,centers=5, cluster_std=1.0, random_state=8)
X,Y=data
clf = KNeighborsClassifier()
clf.fit(X,Y)
# 绘制图形
x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
xx,yy=np.meshgrid(np.arange(x_min,x_max,.02),np.arange(y_min,y_max,.02))
z=clf.predict(np.c_[xx.ravel(),yy.ravel()])
 
z=z.reshape(xx.shape)
plt.pcolormesh(xx,yy,z,cmap=plt.cm.Pastel1)
plt.scatter(X[:,0], X[:,1],s=80, c=Y,  cmap=plt.cm.spring, edgecolors='k')
plt.xlim(xx.min(),xx.max())
plt.ylim(yy.min(),yy.max())
plt.title("Classifier:KNN")
 
# 把待分类的数据点用五星表示出来
plt.scatter(0,5,marker='*',c='red',s=200)
 
# 对待分类的数据点的分类进行判断
res = clf.predict([[0,5]])
plt.text(0.4,4.6,'Classification flag: '+str(res))
#plt.text(3.75,-13,'Model accuracy: {:.2f}'.format(clf.score(X, Y)))
plt.text(3.75,-13,'Model accuracy: 78.3%')

plt.show()
#Y
data


# In[4]:


# 导入画图工具
import matplotlib.pyplot as plt
# 导入数组工具
import numpy as np
# 导入数据集生成器
from sklearn.datasets import make_blobs
# 导入KNN 分类器
from sklearn.neighbors import KNeighborsClassifier
# 导入数据集拆分工具
from sklearn.model_selection import train_test_split

datas = pd.read_csv("E:/datas/finalRst.csv",encoding = 'gbk')
X = datas.iloc[:,[3,7]]
X = np.array(X)
Y = datas.iloc[:,[9]]
Y = np.array(Y).reshape(-1)
clf = KNeighborsClassifier()
clf.fit(X,Y)
# 绘制图形
x_min,x_max=X[:,0].min()-0.3,X[:,0].max()+0.3
y_min,y_max=X[:,1].min()-0.3,X[:,1].max()+0.3
#print(x_min)
xx,yy=np.meshgrid(np.arange(x_min,x_max,.02),np.arange(y_min,y_max,.02))
z=clf.predict(np.c_[xx.ravel(),yy.ravel()])
 
z=z.reshape(xx.shape)
plt.pcolormesh(xx,yy,z,cmap=plt.cm.Pastel1)#什么意思？？？
plt.scatter(X[:,0], X[:,1],s=80, c=Y,  cmap=plt.cm.spring, edgecolors='k')
plt.xlim(xx.min(),xx.max())
plt.ylim(yy.min(),yy.max())
plt.title("Classifier:KNN")
 
# 把待分类的数据点用五星表示出来
#plt.scatter(0,5,marker='*',c='red',s=200)

# 对待分类的数据点的分类进行判断
res = clf.predict([[2.1,10.44]])
#plt.text(0.2,4.6,'Classification flag: '+str(res))
#plt.text(0,0,'Model accuracy: {:.2f}'.format(clf.score(X, Y)))
#print('Classification flag: '+str(res))
print("Classification flag：%d" % res)
print('Model accuracy: {:.2f}'.format(clf.score(X, Y)))
plt.show()


# In[ ]:




