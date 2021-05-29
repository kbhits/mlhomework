'''
实验四
实现感知机学习算法
'''

import numpy as np
from numpy.matrixlib.defmatrix import matrix
import pandas as pd
import matplotlib.pyplot as plt

#data_path = r'C:\Users\ky\Desktop\ml\experience4\perceptron_data.txt'
data_path = r'perceptron_data.txt'
data_title = ['x1', 'x2', 'y']

#绘制图像
def plot(data_path, data_title, w, thita):
    datas = pd.read_csv(data_path, sep='\s+', names=data_title)
    group = datas.groupby('y')
    ggroup = list(group)
    data0 = (np.array(ggroup[0][1])).T
    data1 = (np.array(ggroup[1][1])).T
    y0 = (thita + w[0] * 4) / w[1]
    y1 = (thita - w[0] * 4) / w[1]
    plt.plot([-4, 4], [y0, y1])
    plt.scatter(data0[0], data0[1], c='red')
    plt.scatter(data1[0], data1[1], c='blue')
    plt.show()
    plt.savefig('table.png')

#读取数据
def readData(data_path, data_title):
    datas = pd.read_csv(data_path, sep='\s+', names=data_title)
    data = np.array(datas)
    #将分类为0的标签更改为-1
    for x in data:
        if (x[2] == 0):
            x[2] = -1
    return data

#符号函数
def sign(x):
    if x >= 0:
        return 1
    else:
        return -1

#感知机学习算法
def learn(data, leraning_rate):
    #自定义初始化参数
    w = [1, 1]
    thita = 1
    #同一数据集多次训练
    while(True):
        #学习一个样本
        for x in data:
            sum = 0
            ifchange = 0
            #计算分类结果
            for i in range(len(w)):
                sum = sum + w[i] * x[i]
            sum = sum + (-1) * thita
            y = sign(sum)
            #更新参数
            for i in range(len(w)):
                delta_w = leraning_rate * (x[2] - y) * x[i]
                new_w = w[i] + delta_w
                if new_w != w[i]:
                    ifchange = 1
                w[i] = new_w
            delta_thita = leraning_rate * (x[2] - y) * (-1)
            new_thita = thita + delta_thita
            if new_thita != thita:
                ifchange = 1
            thita = new_thita
            #打印参数
            if ifchange:
                print ("The weight after updating:", end = '')
                for i in range(len(w)):
                    print ("{:.2f}".format(w[i]), end='\t')
                print ("{:.2f}".format(thita))
        #训练完一个样本集，判断参数是否训练达到要求
        arr = 0
        for x in data:
            sum = 0
            for i in range(len(w)):
                sum = sum + w[i] * x[i]
            sum = sum + (-1) * thita
            y = sign(sum)
            if x[2] == y:
                arr = arr + 1
        #如果能将样本集中所有样本分类接近全部正确，停止学习
        if (1 - arr / len(data) < 1e-5):
            break
    return w, thita

if __name__ == '__main__':
    data = readData(data_path, data_title)
    w, thita = learn(data, 0.7)
    plot(data_path, data_title, w, thita)