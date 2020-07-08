import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import h5py
from scipy import io
import matplotlib.pyplot as plt
from os.path import join
from numpy import tile

from skimage import io


def loadNpy():
    testImgs = np.load('./data/test_set_fer2013.npy')
    testLabel = np.load('./data/test_labels_fer2013.npy')

    trainImgs = np.load('./data/data_set_fer2013.npy')
    trainLabel = np.load('./data/data_labels_fer2013.npy')

    testPath = './data/test'
    trainPath = './data/train'

    if not os.path.exists(testPath):
        os.mkdir(testPath)

    if not os.path.exists(trainPath):
        os.mkdir(trainPath)

    for ii in range(7):
        path = join(testPath, str(ii))
        if not os.path.exists(path):
            os.mkdir(path)
        path = join(trainPath, str(ii))
        if not os.path.exists(path):
            os.mkdir(path)

    num = 0
    # write train images
    for ii in range(len(trainImgs)):
        label = trainLabel[ii]
        imgdata = trainImgs[ii]
        img = tile(np.reshape(imgdata, [imgdata.shape[0], imgdata.shape[1], 1]), [3])
        # io.imshow(img)
        spath = '%s/%s/%05d.png' % (trainPath,list(label).index(1), num)
        io.imsave(spath, img)

    # write test images
    for ii in range(len(testImgs)):
        img = testImgs[ii]
        label = testLabel[ii]

    print('end loadNpy()')


def main():
    loadNpy()
    print('end main()')


if __name__ == '__main__':
    main()
