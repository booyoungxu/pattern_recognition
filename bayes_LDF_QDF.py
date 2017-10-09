# -*- coding: utf-8 -*-

import pickle
import os
import numpy as np
from sklearn.decomposition import PCA

filepath = '/home/xu/Projects/pattern_recognition/data'


def prepare_data(path):
    """
    将数据序列化到文件
    :param path: 原数据的路径
    :return: 训练数据和测试数据序列化后的文件
    """
    train_image = []
    train_label = []
    for i in range(1, 6):
        file_path = os.path.join(path, 'cifar-10-batches-py/data_batch_%d' % i)
        with open(file_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            train_image.append(data['data'])
            train_label.append(data['labels'])
    train_image = np.array(train_image).reshape(50000, -1)
    train_label = np.array(train_label).reshape(50000)
    # print(train_images.shape, train_labels.shape)
    with open(os.path.join(path, 'train_images.pkl'), 'wb') as image_train:
        pickle.dump(train_image, image_train)
    with open(os.path.join(path, 'train_labels.pkl'), 'wb') as label_train:
        pickle.dump(train_label, label_train)
    # ss = pickle.load(open(os.path.join(path, 'train_images.pkl'), 'rb'))
    # print(ss.shape)
    file_path = os.path.join(path, 'cifar-10-batches-py/test_batch')
    with open(file_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    test_image = np.array(data['data'])
    test_label = np.array(data['labels'])
    # print(test_images.shape, test_labels.shape)
    with open(os.path.join(path, 'test_images.pkl'), 'wb') as image_test:
        pickle.dump(test_image, image_test)
    with open(os.path.join(path, 'test_labels.pkl'), 'wb') as label_test:
        pickle.dump(test_label, label_test)
    # ss = pickle.load(open(os.path.join(path, 'test_images.pkl'), 'rb'))
    # print(ss.shape)


def pca_feature(train_image, test_image, k):
    """
    对特征向量进行降维
    :param train_image: 训练图片的特征向量
    :param test_image: 测试图片的特征向量
    :param k: pca降维之后的特征维数
    :return: 降维后的训练数据的特征向量和测试数据的特征向量
    """
    pca = PCA(n_components=k, copy=True, whiten=False)
    train_image = pca.fit_transform(train_image)    # 用数据train_image来训练PCA对象， 返回降维后的数据
    test_image = pca.transform(test_image)  # 用train_image训练好的模型对test_image进行降维
    print('explained_variance_ratio_:', sum(pca.explained_variance_ratio_))     # 保留成分的方差百分比之和
    return train_image, test_image


def LDF_2(train_image, train_label, test_image, test_label, prior=None):
    """
    二类线性判别函数, 每类的协方差矩阵相等
    :param train_image: 训练特征
    :param train_label: 训练标签
    :param test_image: 测试特征
    :param test_label: 测试标签
    :param prior: 先验概率
    :return: 测试数据在模型上的精度
    """
    n_samples, n_features = train_image.shape
    class_label, indices = np.unique(train_label, return_inverse=True)  # 排好序的label和每个label在排好序中的class_label的位置
    average = np.zeros(shape=(2, n_features))
    for c in range(2):   # 对每类的特征向量求均值
        train_xc = train_image[indices == c, :]
        average[c] = np.mean(train_xc, axis=0)
    train_2 = [train_image[indices == 0, :], train_image[indices == 1, :]]
    train_2 = np.array(train_2).reshape((-1, 100))
    if prior is None:   # 计算先验概率
        prior = np.bincount(indices) / float(train_2.shape[0])
    variance = np.var(train_2, axis=0)
    for i in range(n_features):
        if variance[i] == 0:
            variance[i] = 1.0e-4
    sigma = np.cov(train_2/variance, rowvar=False)

    if np.linalg.matrix_rank(sigma) < n_features:  # 如果协方差矩阵奇异, 强制缩并
        sigma = (1 - 1.0e-4) * sigma + 1.0e-4 * np.eye(n_features)
    sigma_inv = np.linalg.inv(sigma)

    def g_ci(x, average_c, inv_sigma, prior_c):
        average_c.shape = (np.array(average_c).shape[0], 1)
        w = np.dot(sigma_inv, average_c)
        b = -0.5*np.dot(np.dot(np.transpose(average_c), inv_sigma), average_c) + np.log(prior_c)
        return np.dot(np.transpose(w), x) + b

    # 预测测试样本的标签, 计算正确率
    class_label, indices = np.unique(test_label, return_inverse=True)
    test_2 = [test_image[indices == 0, :], test_image[indices == 1, :]]
    test_2 = np.array(test_2).reshape((-1, 100))
    test_label = [np.zeros(1000), np.zeros(1000)+1]
    test_label = np.array(test_label).reshape((2000, -1))
    n_test_samples = test_2.shape[0]
    g_test = np.zeros(2)
    test_accuracy = 0
    for i in range(n_test_samples):
        test_sample = test_2[i]
        for c in range(2):
            g_test[c] = g_ci(test_sample, average[c], sigma_inv, prior[c])
        if np.where(g_test == max(g_test))[0][0] == test_label[i]:
            test_accuracy += 1
    test_accuracy /= n_test_samples
    return test_accuracy


def LDF(train_image, train_label, test_image, test_label, prior=None):
    """
    十类线性判别函数, 每类的协方差矩阵相等
    :param train_image: 训练特征
    :param train_label: 训练标签
    :param test_image: 测试特征
    :param test_label: 测试标签
    :param prior: 先验概率
    :return: 测试数据在模型上的精度
    """
    n_samples, n_features = train_image.shape
    class_label, indices = np.unique(train_label, return_inverse=True)  # 排好序的label和每个label在排好序中的class_label的位置
    if prior is None:   # 计算先验概率
        prior = np.bincount(indices) / float(n_samples)
    average = np.zeros(shape=(len(class_label), n_features))
    for c in class_label:   # 对每类的特征向量求均值
        train_xc = train_image[indices == c, :]
        average[c] = np.mean(train_xc, axis=0)

    variance = np.var(train_image, axis=0)
    for i in range(n_features):
        if variance[i] == 0:
            variance[i] = 1.0e-4

    sigma = np.cov(train_image, rowvar=False)

    if np.linalg.matrix_rank(sigma) < n_features:  # 如果协方差矩阵奇异, 强制缩并
        sigma = (1 - 1.0e-4) * sigma + 1.0e-4 * np.eye(n_features)
    sigma_inv = np.linalg.inv(sigma)

    def g_ci(x, average_c, inv_sigma, prior_c):
        average_c.shape = (np.array(average_c).shape[0], 1)
        w = np.dot(sigma_inv, average_c)
        b = -0.5*np.dot(np.dot(np.transpose(average_c), inv_sigma), average_c) + np.log(prior_c)
        return np.dot(np.transpose(w), x) + b

    # 预测测试样本的标签, 计算正确率
    test_accuracy = 0
    n_test_samples = test_image.shape[0]
    g_test = np.zeros(len(class_label))
    for i in range(n_test_samples):
        test_sample = test_image[i]
        for c in class_label:
            g_test[c] = g_ci(test_sample, average[c], sigma_inv, prior[c])
        if np.where(g_test == max(g_test))[0][0] == test_label[i]:
            test_accuracy += 1
    test_accuracy /= n_test_samples
    return test_accuracy


def QDF_2(train_image, train_label, test_image, test_label, prior=None):
    """
    二类二次判别，每一类的协方差矩阵不等
    :param train_image: 训练特征
    :param train_label: 训练标签
    :param test_image: 测试特征
    :param test_label: 测试标签
    :param prior: 先验概率
    :return: 测试数据在模型上的精度
    """
    n_samples, n_features = train_image.shape
    class_label, indices = np.unique(train_label, return_inverse=True)  # 排好序的label和每个label在排好序中的class_label的位置
    average = np.zeros(shape=(2, n_features))
    sigma = np.zeros(shape=(2, n_features, n_features))
    sigma_inv = np.zeros(shape=(2, n_features, n_features))
    for c in range(2):  # 对每类的特征向量求均值
        train_xc = train_image[indices == c, :]
        average[c] = np.mean(train_xc, axis=0)
        sigma[c] = np.cov(train_xc, rowvar=False)
        if np.linalg.matrix_rank(sigma[c]) < n_features:  # 如果协方差矩阵奇异, 强制缩并
            sigma[c] = (1 - 1.0e-4) * sigma[c] + 1.0e-4 * np.eye(n_features)
        sigma_inv[c] = np.linalg.inv(sigma[c])
    train_2 = [train_image[indices == 0, :], train_image[indices == 1, :]]
    train_2 = np.array(train_2).reshape((-1, 100))
    if prior is None:  # 计算先验概率
        prior = np.bincount(indices) / float(train_2.shape[0])

    def g_ci(x, average_c, sigma_c, inv_sigma_c, prior_c):
        average_c.shape = (np.array(average_c).shape[0], 1)
        W = -0.5*inv_sigma_c
        w = np.dot(inv_sigma_c, average_c)
        (sign, logdet) = np.linalg.slogdet(sigma_c)
        b = -0.5*np.dot(np.dot(np.transpose(average_c), inv_sigma_c), average_c) - 0.5*(np.log(sign) + logdet) + np.log(prior_c)    # ln|sigma|需要防止overflow
        return np.dot(np.dot(np.transpose(x), W), x) + np.dot(np.transpose(w), x) + b

    # 预测测试样本的标签, 计算正确率
    test_accuracy = 0
    class_label, indices = np.unique(test_label, return_inverse=True)
    test_2 = [test_image[indices == 0, :], test_image[indices == 1, :]]
    test_2 = np.array(test_2).reshape((-1, 100))
    test_label = [np.zeros(1000), np.zeros(1000) + 1]
    test_label = np.array(test_label).reshape((2000, -1))
    n_test_samples = test_2.shape[0]
    g_test = np.zeros(2)
    for i in range(n_test_samples):
        test_sample = test_2[i]
        for c in range(2):
            g_test[c] = g_ci(test_sample, average[c], sigma[c], sigma_inv[c], prior[c])
        if np.where(g_test == max(g_test))[0][0] == test_label[i]:
            test_accuracy += 1
    test_accuracy /= n_test_samples
    return test_accuracy


def QDF(train_image, train_label, test_image, test_label, prior=None):
    """
    十类二次判别，每一类的协方差矩阵不等
    :param train_image: 训练特征
    :param train_label: 训练标签
    :param test_image: 测试特征
    :param test_label: 测试标签
    :param prior: 先验概率
    :return: 测试数据在模型上的精度
    """
    n_samples, n_features = train_image.shape
    class_label, indices = np.unique(train_label, return_inverse=True)  # 排好序的label和每个label在排好序中的class_label的位置
    if prior is None:  # 计算先验概率
        prior = np.bincount(indices) / float(n_samples)
    average = np.zeros(shape=(len(class_label), n_features))
    sigma = np.zeros(shape=(len(class_label), n_features, n_features))
    sigma_inv = np.zeros(shape=(len(class_label), n_features, n_features))
    for c in class_label:  # 对每类的特征向量求均值
        train_xc = train_image[indices == c, :]
        average[c] = np.mean(train_xc, axis=0)
        sigma[c] = np.cov(train_xc, rowvar=False)
        if np.linalg.matrix_rank(sigma[c]) < n_features:  # 如果协方差矩阵奇异, 强制缩并
            sigma[c] = (1 - 1.0e-4) * sigma[c] + 1.0e-4 * np.eye(n_features)
        sigma_inv[c] = np.linalg.inv(sigma[c])

    def g_ci(x, average_c, sigma_c, inv_sigma_c, prior_c):
        average_c.shape = (np.array(average_c).shape[0], 1)
        W = -0.5*inv_sigma_c
        w = np.dot(inv_sigma_c, average_c)
        (sign, logdet) = np.linalg.slogdet(sigma_c)
        b = -0.5*np.dot(np.dot(np.transpose(average_c), inv_sigma_c), average_c) - 0.5*(np.log(sign) + logdet) + np.log(prior_c)    # ln|sigma|需要防止overflow
        return np.dot(np.dot(np.transpose(x), W), x) + np.dot(np.transpose(w), x) + b

    # 预测测试样本的标签, 计算正确率
    test_accuracy = 0
    n_test_samples = test_image.shape[0]
    g_test = np.zeros(len(class_label))
    for i in range(n_test_samples):
        test_sample = test_image[i]
        for c in class_label:
            g_test[c] = g_ci(test_sample, average[c], sigma[c], sigma_inv[c], prior[c])
        if np.where(g_test == max(g_test))[0][0] == test_label[i]:
            test_accuracy += 1
    test_accuracy /= n_test_samples
    return test_accuracy


if __name__ == '__main__':
    prepare_data(filepath)
    train_images = pickle.load(open(os.path.join(filepath, 'train_images.pkl'), 'rb'))
    train_labels = pickle.load(open(os.path.join(filepath, 'train_labels.pkl'), 'rb'))
    test_images = pickle.load(open(os.path.join(filepath, 'test_images.pkl'), 'rb'))
    test_labels = pickle.load(open(os.path.join(filepath, 'test_labels.pkl'), 'rb'))
    train_images, test_images = pca_feature(train_images, test_images, 100)
    accuracy_LDF_2 = LDF_2(train_images, train_labels, test_images, test_labels)
    print('LDF_2: accuracy = %.2f%%' % (100 * accuracy_LDF_2))
    accuracy_LDF = LDF(train_images, train_labels, test_images, test_labels)
    print('LDF: accuracy = %.2f%%' % (100 * accuracy_LDF))
    accuracy_QDF_2 = QDF_2(train_images, train_labels, test_images, test_labels)
    print('QDF_2: accuracy = %.2f%%' % (100 * accuracy_QDF_2))
    accuracy_QDF = QDF(train_images, train_labels, test_images, test_labels)
    print('QDF: accuracy = %.2f%%' % (100 * accuracy_QDF))





