#coding=utf-8
from software import Ui_MainWindow
import sys
import pandas as pd
import matplotlib.pyplot as plt
import csv
from PyQt5.QtWidgets import QApplication,QMainWindow
#from PyQt5.QtWidgets import *
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QFileDialog
from getFeature import getFeatures
from algorithm import trainANDpredict
from getTest import getTestData
from predictTest import predictTests
# from chemical import drawPicture
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from PyQt5.QtWidgets import QHeaderView
from PyQt5.QtWidgets import QTableWidgetItem
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn import model_selection

class WindowShow(QMainWindow,Ui_MainWindow):
    def __init__(self,parent=None):
        super(WindowShow,self).__init__(parent)
        self.setupUi(self)
        # 设置表格为自适应的伸缩模式，即可根据窗口的大小来改变网格的大小
        self.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    # 定义槽函数
    def load_data(self):
        # 打开文件路径
        # 设置文件扩展名过滤,注意用双分号间隔
        fname, ftype = QFileDialog.getOpenFileName(self,'OpenFile',"E:/datas/")
        if len(fname) != 0:
            getTestData(fname)
            self.imagelabel.setPixmap(QtGui.QPixmap("./curve.jpg"))
            self.imagelabel.setScaledContents(True)  # 让图片自适应label大小
            # 利用textBrowser显示文件名
            self.displayFname.setText(fname)

    def load_train(self):
        # 打开文件路径
        # 设置文件扩展名过滤,注意用双分号间隔
        dirname = QFileDialog.getExistingDirectory(self,'OpenDir',"E:/datas/")
        # print(dirname)
        if len(dirname) != 0:
            getFeatures(dirname)
            self.imagelabel_2.setPixmap(QtGui.QPixmap("./all_sample.jpg"))
            self.imagelabel_2.setScaledContents(True)  # 让图片自适应label大小
            # 利用textBrowser显示文件夹名
            self.displayDirname.setText(dirname)

    def getFeature(self):
        # self.displayDirname.setText("正在提取样本特征...")
        if len(self.displayDirname.toPlainText()) != 0:
            root = self.displayDirname.toPlainText()
            filenum = getFeatures(root)
            self.displayDirname.setText("已提取%d条样本特征!" %filenum)

    def train_and_predict(self):
        if len(self.displayDirname.toPlainText()) != 0:
            if self.algorithmType.currentText() == "KNN":
                clfs = KNeighborsClassifier()
            elif self.algorithmType.currentText() == "朴素贝叶斯":
                clfs = GaussianNB()
            elif self.algorithmType.currentText() == "决策树":
                clfs = tree.DecisionTreeClassifier(criterion ='entropy')
            else:
                clfs = svm.SVC(kernel='rbf',gamma=0.01,decision_function_shape='ovo',C=0.8)

            features = []
            if self.CV.isChecked():
                features.append(3)
            if self.waveRate.isChecked():
                features.append(4)
            if self.skewness.isChecked():
                features.append(5)
            if self.kurtosis.isChecked():
                features.append(6)
            if self.peakPos.isChecked():
                features.append(7)

            acc = trainANDpredict(features,clfs)
            self.displayAccuracy.setText("模型准确率为：%.4f" %acc)


    def predict(self):
        # print(self.tableWidget.item(1,1).text())
        if len(self.displayAccuracy.toPlainText()) != 0:
            testRoot = self.displayFname.toPlainText()
            feat, cate = predictTests(testRoot)
            # print(feat)
            if self.algorithmType.currentText() == "KNN":
                clfs = KNeighborsClassifier()
            elif self.algorithmType.currentText() == "朴素贝叶斯":
                clfs = GaussianNB()
            elif self.algorithmType.currentText() == "决策树":
                clfs = tree.DecisionTreeClassifier(criterion='entropy')
            else:
                clfs = svm.SVC(kernel='rbf', gamma=0.01, decision_function_shape='ovo', C=0.8)

            features = []
            if self.CV.isChecked():
                features.append(3)
            if self.waveRate.isChecked():
                features.append(4)
            if self.skewness.isChecked():
                features.append(5)
            if self.kurtosis.isChecked():
                features.append(6)
            if self.peakPos.isChecked():
                features.append(7)

            # 必须是倒序，否则会出错
            if not self.kurtosis.isChecked():
                feat = np.delete(feat, 4)
            if not self.skewness.isChecked():
                feat = np.delete(feat, 3)
            if not self.waveRate.isChecked():
                feat = np.delete(feat, 2)
            if not self.CV.isChecked():
                feat = np.delete(feat, 1)
            if not self.peakPos.isChecked():
                feat = np.delete(feat, 0)
            feat = feat.reshape(1, -1)
            # print(feat)
            # print(features)
            # features存在值即为真
            if features:
                acc = trainANDpredict(features, clfs)
                # print(clfs.predict(feat))
                row = self.tableWidget.rowCount()
                self.tableWidget.insertRow(row)
                # 获取文件名
                filename = self.displayFname.toPlainText()
                index = filename.rfind('/')
                filename = filename[index+1:]
                itemFileName = QTableWidgetItem(filename)
                # 获取预测分类
                itemPredict = QTableWidgetItem(str(clfs.predict(feat)))
                # 获取真实分类
                itemFact = QTableWidgetItem(cate)

                self.tableWidget.setItem(row, 0, itemFileName)
                self.tableWidget.setItem(row, 1, itemPredict)
                self.tableWidget.setItem(row, 2, itemFact)

    def download_file(self):
        # fileName_choose, filetype = QFileDialog.getSaveFileName(self,"文件保存", self.cwd,"All Files (*);;Text Files (*.txt)")
        if self.tableWidget.rowCount() != 0:
            filename, filetype = QFileDialog.getSaveFileName(self, "导出数据", 'C:/', "CSV Files (*.csv)")
            # print(filename)
            if len(filename) != 0:
                with open(filename, 'w') as csvfile:
                    rowCnt = self.tableWidget.rowCount()
                    writer = csv.writer(csvfile)
                    writer.writerow(["文件名", "预测分类", "真实分类"])
                    # 写入多行用writerows
                    for row in range(rowCnt):
                        fnameRow = self.tableWidget.item(row, 0).text()
                        predictRow = self.tableWidget.item(row, 1).text()
                        factRow = self.tableWidget.item(row, 2).text()
                        writer.writerow([fnameRow, predictRow, factRow])


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = WindowShow()
    ui.show()
    sys.exit(app.exec_())
