# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np
from collections import Counter
from scipy.spatial.distance import cosine
import pickle
import os
from sklearn.decomposition import PCA


def pca_feature(train_images, test_images, d):
    """
    对特征向量进行降维
    :param train_images: 训练图片的特征向量
    :param test_images: 测试图片的特征向量
    :param d: pca降维之后的特征维数
    :return: 降维后的训练数据的特征向量和测试数据的特征向量
    """
    pca = PCA(n_components=d, copy=True, whiten=False)
    train_images = pca.fit_transform(train_images)    # 用数据train_image来训练PCA对象， 返回降维后的数据
    test_images = pca.transform(test_images)  # 用train_image训练好的模型对test_image进行降维
    print('explained_variance_ratio_:', sum(pca.explained_variance_ratio_))     # 保留成分的方差百分比之和
    return train_images, test_images


def knn(train_images, train_labels, test_images, test_labels, num):
    """
    KNN进行分类
    :param train_images: 训练集的图片
    :param train_labels: 训练集的标签
    :param test_images: 测试集的图片
    :param test_labels: 测试集的标签
    :param num: 取值k的list
    :return:
    """
    accuracy = np.zeros(len(num))
    test_samples = test_images.shape[0]
    for i in range(test_samples):  # 对于每一个测试样本进行KNN
        x = test_images[i]
        dist = [cosine(x, train_x) for train_x in train_images] # 算测试样本同所有训练数据的cos距离
        index = np.argsort(dist)[0: max(num)] # 取前k小距离的index
        for j in range(len(num)):   # 对于每一个取值k
            y = Counter(train_labels[index[0:num[j]]]).most_common(1)[0][0]
            if y == test_labels[i]:
                accuracy[j] += 1
    accuracy = accuracy/test_samples
    return accuracy

if __name__ == '__main__':
    path = '/home/xu/Projects/pattern_recognition/data/MNIST'
    train_image = pickle.load(open(os.path.join(path, 'train_images.pkl'), 'rb'))
    train_label = pickle.load(open(os.path.join(path, 'train_labels.pkl'), 'rb'))
    test_image = pickle.load(open(os.path.join(path, 'test_images.pkl'), 'rb'))
    test_label = pickle.load(open(os.path.join(path,'test_labels.pkl'), 'rb'))
    print('shape', train_image.shape, train_label.shape, test_image.shape, test_label.shape)
    dimension = [784, 600, 500, 400, 300, 200, 100, 10]
    for dim in dimension:
        train_image_k, test_image_k = pca_feature(train_image, test_image, dim)
        para = [1, 5, 15, 25, 35]
        accuracies = knn(train_image_k, train_label, test_image_k, test_label, para)
        for k in range(len(para)):
            print('knn:   feature=%d,   k=%d,   accuracy = %.2f%%' % (dim, para[k], 100 * accuracies[k]))
    # 对于同一个dim,K=5时精度最高，对于不同的K,K降到100时精度不断提高，当K降为10时，精度降低