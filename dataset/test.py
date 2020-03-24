"""读取mnist数据集"""
import pickle

f = open('mnist.pkl','rb')
info = pickle.load(f)
print(info)