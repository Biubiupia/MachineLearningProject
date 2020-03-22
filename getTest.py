import pandas as pd
import matplotlib.pyplot as plt
def getTestData(filename):
    data = pd.read_csv(filename)
    x = data.iloc[:, :1]
    y = data.iloc[:, 1:]
    # 绘制曲线
    plt.figure()
    plt.plot(x, y, 'k',linestyle=':',marker='*', label="abs")
    plt.xlabel("WL(nm)")
    plt.ylabel('abs')
    plt.title("Curve")
    plt.savefig("curve.jpg")