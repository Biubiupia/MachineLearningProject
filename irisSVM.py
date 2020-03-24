# !/usr/bin/env python
# encoding: utf-8
__author__ = 'Du Hongqing'
from sklearn import svm
import numpy as np
from sklearn import model_selection
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors

def iris_type(s):
    class_label={b'Iris-setosa':0,b'Iris-versicolor':1,b'Iris-virginica':2}
    return class_label[s]


filepath = 'E:/datas/iris.csv'  # 数据文件路径
data=np.loadtxt(filepath,dtype=float,delimiter=',',converters={4:iris_type})
# print(data)
#读入结果示例为：
# [[ 5.1  3.5  1.4  0.2  0. ]
#  [ 4.9  3.   1.4  0.2  0. ]
#  [ 4.7  3.2  1.3  0.2  0. ]
#  [ 4.6  3.1  1.5  0.2  0. ]
#  [ 5.   3.6  1.4  0.2  0. ]]

X, y = np.split(data,(4,),axis=1)
x = X[:,0:2]
x_train,x_test,y_train,y_test=model_selection.train_test_split(x,y,random_state=1,test_size=0.3)

# classifier=svm.SVC(kernel='linear',gamma=0.1,decision_function_shape='ovo',C=0.1)
# kernel='linear'时，为线性核函数，C越大分类效果越好，但有可能会过拟合（defaul C=1）

classifier=svm.SVC(kernel='rbf',gamma=0.1,decision_function_shape='ovo',C=0.8)
# kernel='rbf'（default）时，为高斯核函数，gamma值越小，分类界面越连续；gamma值越大，分类界面越“散”，分类效果越好，但有可能会过拟合。
# decision_function_shape='ovo'时，为one v one分类问题，即将类别两两之间进行划分，用二分类的方法模拟多分类的结果。
# decision_function_shape='ovr'时，为one v rest分类问题，即一个类别与其他类别进行划分。

# 开始训练
classifier.fit(x_train,y_train.ravel())
# 调用ravel()函数将矩阵转变成一维数组
# （ravel()函数与flatten()的区别）
# 两者所要实现的功能是一致的（将多维数组降为一维），
# 两者的区别在于返回拷贝（copy）还是返回视图（view），
# numpy.flatten() 返回一份拷贝，对拷贝所做的修改不会影响（reflects）原始矩阵，
# 而numpy.ravel()返回的是视图（view），会影响（reflects）原始矩阵。


def show_accuracy(y_hat,y_train,str):
    pass

# 计算svm分类器的准确率
print("SVM分类器在训练集上的准确率为：",classifier.score(x_train,y_train)) # 0.838095238095
y_train_pred = classifier.predict(x_train)  # 训练集预测结果
# print(y_train_pred)

print("SVM分类器在测试集的准确率为：",classifier.score(x_test,y_test))  # 0.777777777778
y_test_pred = classifier.predict(x_test)  # 测试集预测结果
# print(y_test_pred)

print("====================================================================================")
# 查看决策函数，decision_function中每一列的值代表该点到各类别的距离
# print('TRAIN decision_function:\n', classifier.decision_function(x_train))
# print('TEST decision_function:\n', classifier.decision_function(x_test))

# 绘制图像
# 1.确定坐标轴范围，x，y轴分别表示两个特征
x1_min, x1_max = x[:, 0].min(), x[:, 0].max()  # 第0列的范围  x[:, 0] "："表示所有行，0表示第1列
x2_min, x2_max = x[:, 1].min(), x[:, 1].max()  # 第1列的范围  x[:, 0] "："表示所有行，1表示第2列
x1, x2 = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j]  # 生成网格采样点（用meshgrid函数生成两个网格矩阵X1和X2）
grid_test = np.stack((x1.flat, x2.flat), axis=1)  # 测试点，再通过stack()函数，axis=1，生成测试点
# flat 将矩阵转变成一维数组.与ravel()的区别：flatten返回的是拷贝
# print("grid_test = \n", grid_test)
grid_hat = classifier.predict(grid_test)       # 预测分类值
# print("grid_hat = \n", grid_hat)
grid_hat = grid_hat.reshape(x1.shape)  # 使之与输入的形状相同
# print("grid_hat = \n", grid_hat)

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
plt.title(u'鸢尾花SVM二特征分类', fontsize=15)
# plt.grid()
plt.show()

# 训练集原始结果&预测结果对比
y_train_hat = classifier.predict(x_train)
y_train_1d = y_train.reshape(-1)
comp = zip(y_train_1d,y_train_hat)  # 用zip把原始结果和预测结果放在一起
print("训练集的原始结果与预测结果：\n", list(comp))

# 测试集原始结果&预测结果对比
y_test_hat = classifier.predict(x_test)
y_test_1d = y_test.reshape(-1)
comp = zip(y_test_1d,y_test_hat)
print("测试集的原始结果与预测结果：\n", list(comp))

# 训练集原始&预测对比（可视化）
plt.figure()
plt.subplot(121)
plt.title(u'原始分类', fontsize=15)
plt.scatter(x_train[:,0],x_train[:,1],c=y_train.reshape(-1),edgecolors='k',s=50)
plt.subplot(122)
plt.title(u'预测分类', fontsize=15)
plt.scatter(x_train[:,0],x_train[:,1],c=y_train_hat.reshape(-1),edgecolors='k',s=50)
plt.show()
