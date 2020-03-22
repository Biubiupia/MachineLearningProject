#coding=utf-8
import pandas as pd
from sklearn import linear_model
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from scipy import stats
import os

# # 归一化
# def MaxMinNormalization(x, Max, Min):
#     x = (x - Min) / (Max - Min);
#     return x


def getFeatures(ROOT):
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
    # ROOT = "E:\datas\chem"
    # 新建一个画布
    plt.figure()
    # plt.title("all_sample")
    fileNum = 0
    cluster = 0
    complex = 0
    partical = 0
    poly = 0
    for root, dirs, files in os.walk(ROOT):
        for file_name in files:
            # print(root)
            fileNum += 1
            file_path = os.path.join(root, file_name)
            # print(file_path)
            data = pd.read_csv(file_path)
            x = data.iloc[:, :1]
            # print(x)
            y = data.iloc[:, 1:]
            # 绘制曲线
            #plt.plot(x, y, 'b', label="abs")

            # 计算那几个特征
            # abs = MaxMinNormalization(y, y.max(), y.min())
            abs = (y - y.min()) / (y.max() - y.min())
            #print(abs)

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

            # 加入数组中
            arrFilename.append(file_name)
            arrVar.append(var)
            arrStd.append(std)
            arrCv.append(cv)
            arrwaveRate.append(waveRate)
            arrSkew.append(skew)
            arrKurtosis.append(kurtosis)
            arrMaxWL.append(maxWL)
            arrPVR.append(PVR)
            if root == os.path.join(ROOT, "cluster"):
                arrCategory.append("cluster")
                plt.subplot(221)
                if cluster == 0:
                    plt.plot(x, y, 'r', label="cluster")
                else:
                    plt.plot(x, y, 'r')
                cluster = cluster + 1
            elif root == os.path.join(ROOT, "complex"):
                arrCategory.append("complex")
                plt.subplot(222)
                if complex == 0:
                    plt.plot(x, y, 'g', label="complex")
                else:
                    plt.plot(x, y, 'g')
                complex = complex + 1
            elif root == os.path.join(ROOT, "partical"):
                arrCategory.append("partical")
                plt.subplot(223)
                if partical == 0:
                    plt.plot(x, y, 'b', label="partical")
                else:
                    plt.plot(x, y, 'b')
                partical = partical + 1
            elif root == os.path.join(ROOT, "poly-disperse"):
                arrCategory.append("poly-disperse")
                plt.subplot(224)
                if poly == 0:
                    plt.plot(x, y, 'y', label="poly-disperse")
                else:
                    plt.plot(x, y, 'y')
                poly = poly + 1
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
    # arrCategory = np.array([0]*arrVar.shape[0])
    rst = pd.DataFrame(
        {'filename': arrFilename, 'var': arrVar, 'std': arrStd, 'cv': arrCv, 'waveRate': arrwaveRate, 'skew': arrSkew, \
         'kurtosis': arrKurtosis, 'peakPos': arrMaxWL, 'PVR': arrPVR, 'category': arrCategory})
    rst.to_csv("./output.csv", index=False)  # 保存在当前文件夹
    # 显示图像
    # plt.legend(loc="upper right")  # 显示图中的标签
    plt.subplot(221)
    #plt.xlabel("WL(nm)")
    #plt.ylabel('abs')
    plt.legend(loc="upper right")
    plt.subplot(222)
    #plt.xlabel("WL(nm)")
    #plt.ylabel('abs')
    plt.legend(loc="upper right")
    plt.subplot(223)
    #plt.xlabel("WL(nm)")
    #plt.ylabel('abs')
    plt.legend(loc="upper right")
    plt.subplot(224)
    #plt.xlabel("WL(nm)")
    #plt.ylabel('abs')
    plt.legend(loc="upper right")
    # plt.title("all_sample")
    plt.savefig("all_sample.jpg")
    return fileNum
