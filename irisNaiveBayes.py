# !/usr/bin/env python
# encoding: utf-8
__author__ = 'Du Hongqing'
from sklearn.naive_bayes import GaussianNB,BernoulliNB
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import model_selection
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def iris_type(s):
    class_label={b'Iris-setosa':0,b'Iris-versicolor':1,b'Iris-virginica':2}
    return class_label[s]

filepath='E:/datas/iris.csv'  # 数据文件路径
data=np.loadtxt(filepath,dtype=float,delimiter=',',converters={4:iris_type})
# print(data)
#读入结果示例为：
# [[ 5.1  3.5  1.4  0.2  0. ]
#  [ 4.9  3.   1.4  0.2  0. ]
#  [ 4.7  3.2  1.3  0.2  0. ]
#  [ 4.6  3.1  1.5  0.2  0. ]
#  [ 5.   3.6  1.4  0.2  0. ]]

X ,y=np.split(data,(4,),axis=1)
x=X[:,0:2]
x_train,x_test,y_train,y_test=model_selection.train_test_split(x,y,random_state=1,test_size=0.3)

# 搭建模型，训练GaussianNaiveBayes分类器
classifier = GaussianNB()
# classifier=BernoulliNB()
# 开始训练
classifier.fit(x_train,y_train.ravel())

print("GaussianNB在训练集上的准确率为：",classifier.score(x_train,y_train))
y_hat=classifier.predict(x_train)
print("GaussianNB在测试集上的准确率为：",classifier.score(x_test,y_test))
y_hat=classifier.predict(x_test)
# GaussianNB-输出训练集的准确率为： 0.809523809524
# GaussianNB-输出测试集的准确率为： 0.755555555556

# # 查看决策函数，可以通过decision_function()实现。decision_function中每一列的值代表距离各类别的距离。
# print('decision_function:\n', classifier.decision_function(x_train))
# print('\npredict:\n', classifier.predict(x_train))

# 绘制图像
# 1.确定坐标轴范围，x，y轴分别表示两个特征
x1_min, x1_max = x[:, 0].min(), x[:, 0].max()  # 第0列的范围
x2_min, x2_max = x[:, 1].min(), x[:, 1].max()  # 第1列的范围
x1, x2 = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j]  # 生成网格采样点
grid_test = np.stack((x1.flat, x2.flat), axis=1)  # 测试点
grid_hat = classifier.predict(grid_test)       # 预测分类值
grid_hat = grid_hat.reshape(x1.shape)  # 使之与输入的形状相同


# 2.指定默认字体
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False
# 3.绘制
cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])

alpha=0.5

plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light) # 预测值的显示
# plt.scatter(x[:, 0], x[:, 1], c=y, edgecolors='k', s=50, cmap=cm_dark)  # 样本
plt.plot(x[:, 0], x[:, 1], 'o', alpha=alpha, color='blue', markeredgecolor='k')
plt.scatter(x_test[:, 0], x_test[:, 1], s=120, facecolors='none', zorder=10)  # 圈中测试集样本
plt.xlabel(u'花萼长度', fontsize=13)
plt.ylabel(u'花萼宽度', fontsize=13)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.title(u'鸢尾花GaussianNB分类结果', fontsize=15)
plt.grid() #显示网格
plt.show()
