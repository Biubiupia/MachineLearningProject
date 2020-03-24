import os
for root, dirs, files in os.walk("E:/datas/chem"):
    for file_name in files:
        file_path = os.path.join(root, file_name)
        print(root)         #返回当前目录下的所有文件名

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
import pandas as pd

datas = pd.read_csv("E:/datas/finalRst.csv", encoding='gbk')
X = datas.iloc[:, [3, 7]]
X = np.array(X)
Y = datas.iloc[:, [9]]
Y = np.array(Y).reshape(-1)
clf = KNeighborsClassifier()
clf.fit(X, Y)
# 绘制图形
x_min, x_max = X[:, 0].min() - 0.3, X[:, 0].max() + 0.3
y_min, y_max = X[:, 1].min() - 0.3, X[:, 1].max() + 0.3
# print(x_min)
xx, yy = np.meshgrid(np.arange(x_min, x_max, .02), np.arange(y_min, y_max, .02))
z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

z = z.reshape(xx.shape)
plt.pcolormesh(xx, yy, z, cmap=plt.cm.Pastel1)
plt.scatter(X[:, 0], X[:, 1], s=80, c=Y, cmap=plt.cm.spring, edgecolors='k')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("Classifier:KNN")

# 把待分类的数据点用五星表示出来
# plt.scatter(0,5,marker='*',c='red',s=200)

# 对待分类的数据点的分类进行判断
res = clf.predict([[2.1, 10.44]])
# plt.text(0.2,4.6,'Classification flag: '+str(res))
# plt.text(0,0,'Model accuracy: {:.2f}'.format(clf.score(X, Y)))
# print('Classification flag: '+str(res))
print("Classification flag：%d" % res)
print('Model accuracy: {:.2f}'.format(clf.score(X, Y)))
plt.show()
