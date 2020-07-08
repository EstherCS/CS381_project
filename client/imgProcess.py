# -*- coding: utf-8 -*-

from skimage import io
import cv2
import os
from os.path import join
import numpy as np


# AF = afraid 憂慮
# AN = angry 憤怒
# DI = disgusted 噁心
# HA = happy 開心
# NE = neutral 無表情
# SA = sad 悲傷
# SU = surprised 驚嚇
def imgProcess():
    path = './KDEF'
    emotion = ['AF', 'AN', 'DI', 'HA', 'NE', 'SA', 'SU']

    savepath = './dataset'
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    for ii in range(len(emotion)):
        sp = join(savepath, str(ii))
        if not os.path.exists(sp):
            os.mkdir(sp)
    fileList = os.listdir(path)
    grey = True
    num = 0
    isResize = True
    for fl in fileList:
        imgPath = join(path, fl)
        imgList = os.listdir(imgPath)
        for il in imgList:
            ipath = join(imgPath, il)
            img = io.imread(ipath, as_grey=grey)
            gap = (img.shape[0] - img.shape[1]) / 2
            if not grey:
                imgcuted = np.uint8(img[gap:gap + img.shape[1], :, :] * 255.0)
            else:
                imgcuted = np.uint8(img[gap:gap + img.shape[1], :] * 255.0)
            if isResize:
                imgcuted = cv2.resize(imgcuted, (128, 128))
            label = il[4:6]
            # try:
            imgsavepath = join(savepath, str(emotion.index(label)), '%05d.png' % num)
            io.imsave(imgsavepath, imgcuted)
            num = num + 1
            # except:
            #     pass
            if num % 100 == 0:
                print(num)
    print('end imgProcess()')


def shuffledata(*arrs):
    # 調用案例 x,y = shuffledata(X,Y)
    arrs = list(arrs)
    for i, arr in enumerate(arrs):
        assert len(arrs[0]) == len(arrs[i])
        arrs[i] = np.array(arr)
    p = np.random.permutation(len(arrs[0]))
    return tuple(arr[p] for arr in arrs)


def loadDate(path):
    filePath = os.listdir(path)
    x, y = [], []
    for fp in filePath:
        imgList = os.listdir(join(path, fp))
        for il in imgList:
            temp = np.zeros(7)
            img = io.imread(join(path, fp, il))
            x.append(np.reshape(img / 255.0, (img.shape[0], img.shape[1], 1)))
            temp[int(fp)] = 1
            y.append(temp)
    print('end loadDate()')
    return shuffledata(np.array(x), np.array(y))


def main():
    # imgProcess()
    loadDate('./train')


if __name__ == '__main__':
    main()
