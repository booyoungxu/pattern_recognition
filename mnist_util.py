# -*- coding: utf-8 -*-

import numpy as np
import struct
import os
import pickle

class DataUtil(object):

    def __init__(self, path=None):
        self._path = path
        self._tag = '>'
        self._twoBytes = 'II'
        self._fourBytes = 'IIII'
        self._pictureBytes = '784B'
        self._labelByte = '1B'
        self._twoBytes2 = self._tag + self._twoBytes
        self._fourBytes2 = self._tag + self._fourBytes
        self._pictureBytes2 = self._tag + self._pictureBytes
        self._labelByte2 = self._tag + self._labelByte

    def get_image(self, file_name, flag='train'):
        file_path = os.path.join(self._path, file_name)
        file = open(file_path, 'rb')  # 以二进制方式打开文件
        buf = file.read()
        file.close()
        index = 0
        magic, images, rows, columns = struct.unpack_from(self._fourBytes2, buf, index)  # 从buf的第index个字节开始以大端序读取4个unsigned int
        index += struct.calcsize(self._fourBytes)   # 计算四字节格式所占的内存字节
        image = []
        for i in range(images):
            img = struct.unpack_from(self._pictureBytes2, buf, index)    # 从buf的第index个字节开始以大端序读728个unsigned char
            index += struct.calcsize(self._pictureBytes2)   # 下一张图片的index
            img = list(img)
            for j in range(len(img)):
                if img[j] > 1:
                    img[j] = 1
            image.append(img)
        length = np.array(image).shape[0]
        print('images', np.array(image).shape)
        with open(os.path.join(self._path, flag+'_images.pkl'), 'wb') as out_images:
            pickle.dump(np.array(image), out_images)

    def get_label(self, file_name, flag='train'):
        file_path = os.path.join(self._path, file_name)
        file = open(file_path, 'rb')
        buf = file.read()
        file.close()
        index = 0
        magic, items = struct.unpack_from(self._twoBytes2, buf, index)  # 从buf的第index个字节开始以大端序读取2个unsigned int
        index += struct.calcsize(self._twoBytes2)
        labels = []
        for x in range(items):
            label = struct.unpack_from(self._labelByte2, buf, index)    # 从buf的第index个字节开始以大端序读1个unsigned char
            index += struct.calcsize(self._labelByte2)
            labels.append(label[0])
        print('labels', np.array(labels).shape)
        with open(os.path.join(self._path, flag+'_labels.pkl'), 'wb') as out_labels:
            pickle.dump(np.array(labels), out_labels)


if __name__ == '__main__':
    path = '/home/xu/Projects/pattern_recognition/data/MNIST'
    data = DataUtil(path)
    data.get_image('train-images.idx3-ubyte', 'train')
    data.get_image('t10k-images.idx3-ubyte', 'test')
    data.get_label('train-labels.idx1-ubyte', 'train')
    data.get_label('t10k-labels.idx1-ubyte', 'test')
    # ss = pickle.load(open(os.path.join(path, 'test_labels.pkl'), 'rb'))
    # print('bbnnn', ss.shape)