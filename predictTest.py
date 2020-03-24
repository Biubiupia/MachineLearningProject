#coding=utf-8
import pandas as pd
from sklearn import linear_model
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from scipy import stats
import os

def predictTests(ROOT):
    arrwaveRate = []
    arrSkew = []
    arrKurtosis = []
    # arrCategory = []
    arrMaxWL = []
    arrPVR = []

    data = pd.read_csv(ROOT)
    x = data.iloc[:, :1]
    y = data.iloc[:, 1:]
    abs = (y - y.min()) / (y.max() - y.min())
    # print(abs)

    var = np.var(abs)
    # print("方差为：%f" % var)
    std = np.std(abs, ddof=1)
    # print("标准差为：%f" % std)
    cv = std / np.mean(abs)
    # print("变异系数为：%f" % cv)
    # 计算对数收益率
    # logreturns = diff(log(abs.values.reshape(-1)+0.01))
    # print("变异系数为：%f" % logreturns)
    logreturns = diff(log(abs.values.reshape(-1) + 0.01))
    # print(logreturns)
    waveRate = np.std(logreturns, ddof=1) / np.mean(logreturns)
    waveRate = waveRate / sqrt(1 / abs.shape[0])
    # print("波动率为：", waveRate)
    # 偏度代表性不强
    skew = stats.skew(abs)
    # print("偏度为：%f" % skew)
    kurtosis = stats.kurtosis(abs)
    # print("峰度为：%f" % kurtosis)
    maxIndex = abs.idxmax(axis=0)
    maxWL = np.array(x)[maxIndex]
    # print("文件名：%s" % file_name,"最大值索引为：%d" % maxIndex,"最大值所在波长:%d" % maxWL)
    # 要用归一化前的数据计算峰谷比
    peak = y.max()
    valley = y.min() + 0.01
    PVR = peak / valley
    # print("峰谷比为：%f" % PVR)
    if "cluster" in ROOT:
        category = "cluster"
    elif "complex" in ROOT:
        category = "complex"
    elif "partical" in ROOT:
        category = "partical"
    elif "poly-disperse" in ROOT:
        category = "poly-disperse"
    else:
        category = "unknown"

    arrwaveRate.append(waveRate)
    arrSkew.append(skew)
    arrKurtosis.append(kurtosis)
    arrMaxWL.append(maxWL)
    arrPVR.append(cv)

    arrwaveRate = np.array(arrwaveRate).reshape(-1)
    arrSkew = np.array(arrSkew).reshape(-1)
    arrKurtosis = np.array(arrKurtosis).reshape(-1)
    arrMaxWL = np.array(arrMaxWL).reshape(-1)
    arrPVR = np.array(arrPVR).reshape(-1)

    rst = pd.DataFrame(
        {'peakPos': arrMaxWL, 'PVR': arrPVR,'waveRate': arrwaveRate, 'skew': arrSkew, 'kurtosis': arrKurtosis})
    testFeature = np.array(rst).reshape(-1)
    return testFeature, category
