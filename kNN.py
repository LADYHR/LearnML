# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 15:10:25 2018

@author: Hua Rui
"""

import sys
import os
from numpy import *
import numpy as np
import matplotlib.pyplot as plt
import operator
import importlib

#配置UTF-8输出环境
importlib.reload(sys)

k=2

#夹角余弦公式
def cosdist(vector1,vector2):
    return dot(vector1,vector2)/(linalg.norm(vector1)*linalg.norm(vector2))

#kNN分类器
def classify(testdata, trainSet, ListClasses, k):
    dataSetSize = trainSet.shape[0]
    distances = array(zeros(dataSetSize))
    for indx in range(dataSetSize):
        distances[indx] = cosdist(testdata, trainSet[indx])
    sortedDistIndicies = argsort(-distances)
    classCount = {}
    for i in range(k):
        voteIlabel = ListClasses[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]
        
#
def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0.1,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

#绘制图形
dataSet, labels = createDataSet()
fig = plt.figure()
ax = fig.add_subplot(111)
indx = 0
for point in dataSet:
    if labels[indx] == 'A':
        ax.scatter(point[0],point[1],c='blue',marker='o',linewidths=0,s=300)
        plt.annotate("("+str(point[0])+","+str(point[1])+")",xy=(point[0],point[1]))
    else:
        ax.scatter(point[0],point[1],c='red',marker='^',linewidths=1,s=300)
        plt.annotate("("+str(point[0])+","+str(point[1])+")",xy=(point[0],point[1]))
    indx += 1

testdata = [0.2,0.2]
ax.scatter(testdata[0],testdata[1],c='green',marker='^',linewidths=0,s=300)
plt.annotate("("+str(testdata[0])+","+str(testdata[1])+")",xy=(testdata[0],testdata[1]))
    
plt.show()

testDataLabel = classify(testdata,dataSet,labels,3)
print(testDataLabel)
    
