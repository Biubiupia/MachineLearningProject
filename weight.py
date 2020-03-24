import pandas as pd
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    data = pd.read_csv("E:/datas/weight.csv", header=None, names=['height', 'weight'])

    """# shape 可以得到行数和列数
    print(data.shape)
    print(data.columns)
    print(data.index)
    print(data.head())
    print(data.tail())
    print(data)"""
    # print(data.height[6])
    x_train = np.array(data.height).reshape(-1)
    x_train = np.reshape(x_train, newshape=(20,1))
    y_train = np.array(data.weight).reshape(-1)
    y_train = np.reshape(y_train, newshape=(20, 1))
    x_test = [178,160,190]
    x_test = np.reshape(x_test,newshape=(3,1))

    linear = linear_model.LinearRegression()
    # 训练X和Y
    linear.fit(x_train, y_train)

    # 计算方差分数（1是完美的预测）
    print("方差分数：",linear.score(x_train,y_train))
    # 系数和截距
    print("系数：",linear.coef_)
    print("截距：",linear.intercept_)
    # 预测
    predict = np.array(linear.predict(x_test)).reshape(-1)
    print("预测结果为：",predict)

    # 描点
    plt.scatter(x_train, y_train, color='black')
    # 画直线
    x = np.random.randint(160,200,1000)
    y = np.array(linear.coef_* x + linear.intercept_).reshape(-1)
    plt.plot(x, y, color='blue', linewidth=3)
    # plt.xticks()
    # plt.yticks()
    plt.show()