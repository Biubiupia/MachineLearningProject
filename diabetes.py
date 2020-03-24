import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

#加载糖尿病数据集
diabetes = datasets.load_diabetes()


#只使用一个特性
diabetes_X = diabetes.data[:, np.newaxis, 2]

# 将数据分为训练diabetes_X_train /测试集diabetes_X_test
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# 把目标分成训练diabetes_y_train /测试集diabetes_y_test
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

#创建线性回归对象
regr = linear_model.LinearRegression()

#使用训练集训练模型
regr.fit(diabetes_X_train, diabetes_y_train)

#使用测试集进行预测
diabetes_y_pred = regr.predict(diabetes_X_test)
print("预测结果如下")
print(diabetes_y_pred)

# 查看系数
print('Coefficients: \n', regr.coef_)
# 查看均方误差
print("Mean squared error: %.2f"
      % mean_squared_error(diabetes_y_test, diabetes_y_pred))
# 解释方差分数:1是完美的预测
print('Variance score: %.2f' % r2_score(diabetes_y_test, diabetes_y_pred))

#图输出
plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
#plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()