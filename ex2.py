'''
实验二 LDA
鲜血中心数据集
'''

import math
import numpy as np
from numpy.matrixlib.defmatrix import matrix
import pandas as pd
import matplotlib.pyplot as plt

#数据存储的路径、属性的名称、训练集大小、测试集大小、属性值的数目
#data_path = r'C:\Users\ky\Desktop\ml\experience2\blood_data.txt'
data_path = r'mlhonemwork\blood_data.txt'
data_title = ['Recency (months)', 'Frequency (times)', 'Monetary (c.c. blood)', 'Time (months)', 'whether he/she donated blood in March 2007']
result_title = 'whether he/she donated blood in March 2007'
attribute_num = 4
flag = -1


#读取实验数据，并分成训练集train_data和测试集test_data
def readData(data_path, data_title):
    data = pd.read_csv(data_path, sep=',')
    test_data = pd.concat([data[0:75], data[673:748]])
    train_data = data[75:673]
    return train_data, test_data

#数据处理
def DataProcess(data):
    #分类，分成0、1两类
    group = data.groupby(result_title)
    ggroup = list(group)
    class0 = ggroup[0]
    class1 = ggroup[1]
    #转化成array并删除最后一列属性
    class0_attribute = np.array(class0[1])
    class1_attribute = np.array(class1[1])
    class0_attribute = np.delete(class0_attribute, attribute_num, axis=1)
    class1_attribute = np.delete(class1_attribute, attribute_num, axis=1)
    return class0_attribute, class1_attribute

#计算协方差矩阵
def covariance_matrix(class_attribute, expect, class_num):
    mid_result = class_attribute - expect
    matrix = 0
    for i in list(range(class_num)):
        elem = np.array([mid_result[i]])
        matrix = matrix + np.dot(elem.T, elem)
    return matrix

def train(class0_attribute, class1_attribute):
    global flag
    #求均值向量miu，即期望expect
    class0_num = class0_attribute.shape[0]
    class1_num = class1_attribute.shape[0]
    array0 = np.ones((1, class0_num))
    array1 = np.ones((1, class1_num))
    sum_result0 = np.dot(array0, class0_attribute)
    sum_result1 = np.dot(array1, class1_attribute)
    expect0 = sum_result0 / class0_num
    expect1 = sum_result1 / class1_num
    #求类内散度矩阵
    matrix1 = covariance_matrix(class0_attribute, expect0, class0_num)
    matrix2 = covariance_matrix(class1_attribute, expect1, class1_num)
    scatter_matrix = (matrix1 + matrix2)
    #print(scatter_matrix)
    #求结果w，并将中心点投影到w上求出分类标准
    mid_result = np.linalg.inv(scatter_matrix)
    result = np.dot(mid_result, (expect0 - expect1).T)
    standard = (np.dot(expect0, result) + np.dot(expect1, result))/ 2
    #test_result函数中判断使用
    if np.dot(expect0, result) > np.dot(expect1, result):
        flag = 1
    else:
        flag = 0
    return result, standard

#测试，data_set是测试的数据集、w是上述求解结果、stardard是判断标准，type是该数据集的类别
def test_result(data_set, w, stardard, type):
    stardard_value = stardard[0][0]
    set_num = data_set.shape[0]
    arr = 0
    para = np.array(w)
    right_list = list()
    wrong_list = list()
    for i in list(range(set_num)):
        elem = np.array([data_set[i]])
        result = np.dot(para.T, elem.T)
        result_value = result[0][0]
        #在投影到w上，0类在1类右侧，测试点在标准右侧被归类到0类
        if flag == 1:
            if type == 0 and result_value > stardard_value:
                arr = arr + 1
                right_list.append(result_value)
            elif type == 0:
                wrong_list.append(result_value)
            if type == 1 and result_value <= stardard_value:
                arr = arr + 1
                right_list.append(result_value)
            elif type == 1:
                wrong_list.append(result_value)
        #在投影到w上，0类在1类左侧，测试点在标准右侧被归类到1类
        if flag == 0:
            if type == 0 and result_value < stardard:
                arr = arr + 1
                right_list.append(result_value)
            elif type == 0:
                wrong_list.append(result_value)
            if type == 1 and result_value >= stardard:
                arr = arr + 1
                right_list.append(result_value)
            elif type == 1:
                wrong_list.append(result_value)
    return arr, set_num, right_list, wrong_list

def test(test0, test1, w, stardard):
    arr1, num1, list11, list12 = test_result(test0, w, stardard, 0)
    arr2, num2, list21, list22 = test_result(test1, w, stardard, 1)
    list11.extend(list21)
    list12.extend(list22)
    plt.scatter (list(range(len(list11))), list11, c = 'blue')
    plt.scatter (list(i* 2.5 for i in range(len(list12))), list12, c = 'red')
    plt.plot ([0, len(list11)], [stardard[0][0], stardard[0][0]], c = 'green')
    print ("The number of right test point is ", end='')
    print (arr1 + arr2)
    print ("The accuracy of result is ", end='')
    print ((arr1 + arr2) / (num1 + num2))
    plt.show()
    return 0

if __name__ == '__main__':
    train_data, test_data = readData(data_path, data_title)
    class0_attribute, class1_attribute = DataProcess(train_data)
    test0, test1 = DataProcess(test_data)
    w, stardard = train(class0_attribute, class1_attribute)
    print ("The result of w:")
    print (w)
    test(test0, test1, w, stardard)