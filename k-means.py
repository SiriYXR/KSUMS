#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author:siriyang
@file: k-means.py
@time: 2020/11/28 19:27 
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def randCent(dataMat, k):
    '''
    为给定数据集构建一个包含K个随机质心的集合,
    随机质心必须要在整个数据集的边界之内,这可以通过找到数据集每一维的最小和最大值来完成
    然后生成0到1.0之间的随机数并通过取值范围和最小值,以便确保随机点在数据的边界之内
    :param dataMat:
    :param k:
    :return:
    '''
    # 获取样本数与特征值
    m, n = np.shape(dataMat)
    # 初始化质心,创建(k,n)个以零填充的矩阵
    centroids = np.mat(np.zeros((k, n)))
    # 循环遍历特征值
    for j in range(n):
        # 计算每一列的最小值
        minJ = min(dataMat[:, j])
        # 计算每一列的范围值
        rangeJ = float(max(dataMat[:, j]) - minJ)
        # 计算每一列的质心,并将值赋给centroids
        centroids[:, j] = np.mat(minJ + rangeJ * np.random.rand(k, 1))
    # 返回质心
    return centroids

def distEclud(vecA, vecB):
    '''
    欧氏距离计算函数
    :param vecA:
    :param vecB:
    :return:
    '''
    return np.sqrt(np.sum(np.power(vecA - vecB, 2)))

def cost(dataMat, k, distMeas=distEclud, createCent=randCent,iterNum=300):
    '''
    计算误差的多少，通过这个方法来确定 k 为多少比较合适，这个其实就是一个简化版的 kMeans
    :param dataMat: 数据集
    :param k: 簇的数目
    :param distMeans: 计算距离
    :param createCent: 创建初始质心
    :param iterNum：默认迭代次数
    :return:
    '''
    # 获取样本数和特征数
    m, n = np.shape(dataMat)
    # 初始化一个矩阵来存储每个点的簇分配结果
    # clusterAssment包含两个列:一列记录簇索引值,第二列存储误差(误差是指当前点到簇质心的距离,后面会使用该误差来评价聚类的效果)
    clusterAssment = np.mat(np.zeros((m, 2)))
    # 创建质心,随机K个质心
    centroids = createCent(dataMat, k)
    clusterChanged = True
    while iterNum > 0:
        clusterChanged = False
        # 遍历所有数据找到距离每个点最近的质心,
        # 可以通过对每个点遍历所有质心并计算点到每个质心的距离来完成
        for i in range(m):
            minDist = np.inf
            minIndex = -1
            for j in range(k):
                # 计算数据点到质心的距离
                # 计算距离是使用distMeas参数给出的距离公式,默认距离函数是distEclud
                distJI = distMeas(centroids[j, :], dataMat[i, :])
                # 如果距离比minDist(最小距离)还小,更新minDist(最小距离)和最小质心的index(索引)
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            # 更新簇分配结果为最小质心的index(索引),minDist(最小距离)的平方
            clusterAssment[i, :] = minIndex, minDist ** 2
            iterNum -= 1;
        # print(centroids)
        # 遍历所有质心并更新它们的取值
        for cent in range(k):
            # 通过数据过滤来获得给定簇的所有点
            ptsInClust = dataMat[np.nonzero(clusterAssment[:, 0].A == cent)[0]]
            # 计算所有点的均值,axis=0表示沿矩阵的列方向进行均值计算
            centroids[cent, :] = np.mean(ptsInClust, axis=0)
    # 返回给定迭代次数后误差的值
    return np.mat(clusterAssment[:,1].sum(0))[0,0]

def kMeans(dataMat, k, distMeas=distEclud, createCent=randCent):
    '''
    创建K个质心,然后将每个点分配到最近的质心,再重新计算质心。
    这个过程重复数次,直到数据点的簇分配结果不再改变为止
    :param dataMat: 数据集
    :param k: 簇的数目
    :param distMeans: 计算距离
    :param createCent: 创建初始质心
    :return:
    '''
    # 获取样本数和特征数
    m, n = np.shape(dataMat)
    # 初始化一个矩阵来存储每个点的簇分配结果
    # clusterAssment包含两个列:一列记录簇索引值,第二列存储误差(误差是指当前点到簇质心的距离,后面会使用该误差来评价聚类的效果)
    clusterAssment = np.mat(np.zeros((m, 2)))
    # 创建质心,随机K个质心
    centroids = createCent(dataMat, k)

    # 初始化标志变量,用于判断迭代是否继续,如果True,则继续迭代
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        # 遍历所有数据找到距离每个点最近的质心,
        # 可以通过对每个点遍历所有质心并计算点到每个质心的距离来完成
        for i in range(m):
            minDist = np.inf # 初始化为无穷大
            minIndex = -1
            for j in range(k):
                # 计算数据点到质心的距离
                # 计算距离是使用distMeas参数给出的距离公式,默认距离函数是distEclud
                distJI = distMeas(centroids[j, :], dataMat[i, :])
                # 如果距离比minDist(最小距离)还小,更新minDist(最小距离)和最小质心的index(索引)
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            # 如果任一点的簇分配结果发生改变,则更新clusterChanged标志
            if clusterAssment[i, 0] != minIndex: clusterChanged = True
            # 更新簇分配结果为最小质心的index(索引),minDist(最小距离)的平方
            clusterAssment[i, :] = minIndex, minDist ** 2
        # print(centroids)
        # 遍历所有质心并更新它们的取值
        for cent in range(k):
            # 通过数据过滤来获得给定簇的所有点
            ptsInClust = dataMat[np.nonzero(clusterAssment[:, 0].A == cent)[0]]
            # 计算所有点的均值,axis=0表示沿矩阵的列方向进行均值计算
            centroids[cent, :] = np.mean(ptsInClust, axis=0)
    # 返回所有的类质心与点分配结果
    return centroids, clusterAssment

if __name__ == "__main__":
    df=pd.read_csv("./data/data1.csv")
    data=df.values
    dataMat=np.matrix(data)
    k=3
    res1,res2=kMeans(dataMat,k)

    for i in range(k):
        x=[]
        y=[]
        for j in range(len(df)):
            if res2[j,0]==i:
                x.append(data[j,0])
                y.append(data[j,1])
        plt.scatter(x, y)

        # plt.scatter(res1[i,0],res1[i,1], marker='^',c = 'r',s=300)

    plt.show()

    # df['Y']=res2[:,0].astype(np.int32)
    # print(df)
    # df.to_csv('./tmp.csv',index=False)