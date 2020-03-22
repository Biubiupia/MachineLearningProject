#coding=utf-8
import matplotlib.pyplot as plt
import numpy as np
# 导入数据集拆分工具
from sklearn import model_selection
import pandas as pd
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def trainANDpredict(feature,clf):
    datas = pd.read_csv("./output2.28.csv")
    X = datas.iloc[:,feature]
    X = np.array(X)
    Y = datas.iloc[:,[9]]
    Y = np.array(Y).reshape(-1)
    X_train,X_test,y_train,y_test=model_selection.train_test_split(X,Y,test_size=0.1,random_state=1)

    clf.fit(X_train,y_train)
    #使用测试集数据检验模型准确率
    Accuracy = clf.score(X_test,y_test)
    return Accuracy


