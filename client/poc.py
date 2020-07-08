# Proof-of-concept
# -*- coding: UTF-8 -*-

# import 為 import整個package
# from~import 為 從某個package中import 某個工具
# import~as 為將import進來的package改名
# https://medium.com/pyladies-taiwan/python-%E7%9A%84-import-%E9%99%B7%E9%98%B1-3538e74f57e3
import cv2
import pandas as pd  # 抓取資料用
from matplotlib import pyplot as plt # 繪圖庫
# Matplotlib 是完整的 python package
# pyplot 是 Matplotlib 其中一個 module
import matplotlib
from PIL import Image # 影像處理套件
import sys # 要使用command line可用
from constants import *
from emotion_recognition import EmotionRecognition # 自己寫的py，如同c++中main用到其他cpp檔一樣
from crawler import crawler
import numpy as np # 主要用於資料處理上，能快速操作多重維度的陣列
from pymongo import MongoClient  # python mongoDB
import urllib.request # 與網頁操作有關，例如可以直接連到google
from dataset_loader import DatasetLoader
from os.path import join # 要用路徑時使用
from sklearn.metrics import classification_report, recall_score, precision_score, accuracy_score, f1_score, \
    jaccard_similarity_score
# 評估model、test好不好
# import sys
# sys.path.append('./data')
from cvs_to_numpy import loadData
import string
cascade_classifier = cv2.CascadeClassifier("C:\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml")
from imgProcess import loadDate
from pymongo import MongoClient

connection = MongoClient('140.138.145.77', 27017)
connection.admin.authenticate("bigmms", "bigmms1413b")
tdb = connection.open
post = tdb.record
#db = connection['open']['record']

def get_time():
    from datetime import datetime
    get_time = datetime.now().strftime('%Y%m%d')
    return int(get_time)  #return type int

def insert_mongo(label):
    for ii in post.find():
        print(ii)
    new_label = {"emotion" : label, "time" : get_time()}
    post.insert(new_label)
    #return db.insert_one(new_account).inserted_id

def emotion_to_vec(x):
    d = np.zeros(len(EMOTIONS))
    d[x] = 1.0
    return d

def data_to_image(data):
    # print data
    data_image = np.fromstring(str(data), dtype=np.uint8, sep=' ').reshape((SIZE_FACE, SIZE_FACE))
    data_image = Image.fromarray(data_image).convert('RGB')
    data_image = np.array(data_image)[:, :, ::-1].copy()
    data_image = format_image(data_image)
    return data_image

def brighten(data, b):
    datab = data * b
    return datab

def format_image(image):
    if len(image.shape) > 2 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image = cv2.imdecode(image, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    faces = cascade_classifier.detectMultiScale(
            image,
            scaleFactor=1.3,
            minNeighbors=5
    )
    # None is we don't found an image
    if not len(faces) > 0:
        return None
    max_area_face = faces[0]
    for face in faces:
        if face[2] * face[3] > max_area_face[2] * max_area_face[3]:
            max_area_face = face
    # Chop image to face
    face = max_area_face
    image = image[face[1]:(face[1] + face[2]), face[0]:(face[0] + face[3])]
    # Resize image to network size
    try:
        image = cv2.resize(image, (SIZE_FACE, SIZE_FACE), interpolation=cv2.INTER_CUBIC) / 255.
        cv2.imshow("catchFACE", image)
    except Exception:
        print("[+] Problem during resize")
        return None
        # cv2.imshow("Lol", image)
        # cv2.waitKey(0)
    return image

def showHistory(history=None):
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train','val'], loc='upper left')
    plt.savefig('./accuracy.png')
    # plt.show()
    # summarize history for loss
    plt.clf()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train','val'], loc='upper left')
    plt.savefig('./loss.png')
    # plt.show()

def train():
    # Load Model
    network = EmotionRecognition()
    # network.model.summary()
    # images, labels, images_testing, labels_testing = loadData()
    images, labels = loadDate('./emotionDataset/train')

    history = network.start_training(images, labels)
    network.save_model()
    showHistory(history=history)
    print('train end!')
    # network.build_network()
    # network.load_model2()
    # dataset = DatasetLoader()
    # dataset.load_from_save()
    print('[+] Dataset found and loaded')

def test():
    # images, labels, images_testing, labels_testing = loadData()
    images_testing, labels_testing = loadDate('./emotionDataset/test')
    network = EmotionRecognition()
    # network.build_network()
    network.load_model2()
    print('[+] Testing load model')
    result = network.predict(images_testing)

    # num = 0
    y_true = []
    y_pred = []
    for label in range(7):
        for ii in range(len(result)):
            pre = list(result[ii]).index(np.max(result[ii]))
            gt = list(labels_testing[ii]).index(np.max(labels_testing[ii]))
            # y_true.append(gt)
            # y_pred.append(pre)
        # target_names = ['class 0', 'class 1', 'class 2', 'class 3', 'class 4', 'class 5', 'class 6']
        # print(classification_report(y_true, y_pred, target_names=target_names))
            if gt == label:
                y_true.append(1)
            else:
                y_true.append(0)
            if pre == label:
                y_pred.append(1)
            else:
                y_pred.append(0)


        recall = recall_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        f1_measure = f1_score(y_true, y_pred)
        similarity = jaccard_similarity_score(y_true, y_pred)
        print('###label :%d' % label)
        print('recall:%s    precision:%s    similarity:%s   F1_measure:%s   accuracy:%s' % (recall, precision, similarity, f1_measure, accuracy))
        print("=========================================================================")
        y_true = []
        y_pred = []

def detect():
    video_capture = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX

    feelings_faces = []
    for index, emotion in enumerate(EMOTIONS):
        feelings_faces.append(cv2.imread('./emojis/' + emotion + '.png', -1))

    voting_1 = []
    cal_1 = []
    voting_2 = []
    cal_2 = []
    count = 0
    network = EmotionRecognition()
    # network.build_network()
    network.load_model2()
    print('[+] Testing load model')
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        #cv2.imshow('Video', frame)
        cv2.imwrite("%05d.jpg",frame)
        # Predict result with network
        result = network.predict(format_image(frame))
        # Draw face in frame
        # for (x,y,w,h) in faces:
        #   cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
        # Write results in frame


        if result is not None:
            for index, emotion in enumerate(EMOTIONS):  # 3 顯示每一frame之偵測資訊(文字、直方圖)
                cv2.putText(frame, emotion, (10, index * 20 + 20), cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 0), 1)
                cv2.rectangle(frame, (130, index * 20 + 10), (130 + int(result[0][index] * 100), (index + 1) * 20 + 4),(255, 0, 0), -1)
            count += 1
            print(result[0].tolist().index(max(result[0])))
            voting_1.append(result[0].tolist().index(max(result[0])))
            if len(voting_1) == 60:
                for i in range(7):
                    cal_1.append(voting_1.count(i))
                maximum_face_times = np.max(cal_1)
                maximum_face = cal_1.index(maximum_face_times)
                # print(maximum_face, maximum_face_times)
                voting_2.append(maximum_face)

                voting_1.clear()
                cal_1.clear()

            if (len(voting_2) == 7):  # 之後改次數--!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                for i in range(7):
                    cal_2.append(voting_2.count(i))
                maximum_face_times_2 = np.max(cal_2)
                maximum_face_2 = cal_2.index(maximum_face_times_2)
                print("Voting 結果︰第", maximum_face_2, "類")
                face_image = feelings_faces[maximum_face_2]

                # emotion = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']
                for c in range(0, 3):
                    frame[200:320, 10:130, c] = face_image[:, :, c] * (face_image[:, :, 3] / 255.0) + frame[200:320,10:130, c] * (1.0 - face_image[:, :,3] / 255.0)

                voting_2.clear()
                cal_2.clear()

                ###response
                from pygame import mixer
                import random
                mixer.init()
                mixer.music.load(join('./response/', str(maximum_face_2)) + '/' + str(random.randint(0, 3)) + '.mp3')
                mixer.music.play()

                insert_mongo(int(maximum_face_2))

                # connection = MongoClient('140.138.145.77', 27017)
                # connection.admin.authenticate("bigmms", "bigmms1413b")
                # tdb = connection.musicky
                # post = tdb.test
                #
                # # for i in tdb.test.find({"emotion": EMOTIONS[maximum_face_2]}): print(i)
                # pipeline = [{"$match": {"emotion": EMOTIONS[maximum_face_2]}},
                #             {"$sample": {"size": 1}}]  # 隨機取出一個"emotion":"sad"的資料
                # data = list(tdb.test.aggregate(pipeline))
                # print(data)
                #
                # a = str(data)
                #
                # delete = ["[", "{", "}", "]", "\'"]  # 刪除不必要的符號
                # for i in range(len(delete)):
                #     a = a.replace(delete[i], "")
                #
                # replace = [": ", ", "]  # 替換不必要的符號
                # for j in range(len(replace)):
                #     a = a.replace(replace[j], ",")
                #
                # a = a.split(",")  # 以逗號區分不同字串
                # rand_keyword = a[a.index("keyword") + 1]  # 根據不同的情緒，抓出所要使用的keyword
                # print(rand_keyword)
                # keyword = rand_keyword
                # keyword = urllib.parse.quote(rand_keyword)
                # url = "https://www.youtube.com/results?search_query=" + keyword
                # crawler(url)

        if (count >= 420):  # and (count%180)<=20):    # Ugly transparent fix  # 表情圖片顯示於螢幕 (停留20 frames)
            for c in range(0, 3):
                frame[200:320, 10:130, c] = face_image[:, :, c] * (face_image[:, :, 3] / 255.0) + frame[200:320, 10:130,c] * (1.0 - face_image[:, :,3] / 255.0)
        # # Display the resulting frame
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    video_capture.release()
    cv2.destroyAllWindows()

def main():
    # train()
    # test()
    detect()

if __name__ == '__main__':
    main()

    # [+] Testing load model
    #              precision    recall  f1-score   support
    #
    #     class 0       0.42      0.43      0.42       780
    #     class 1       0.72      0.45      0.56        93
    #     class 2       0.42      0.39      0.40       883
    #     class 3       0.73      0.75      0.74      1586
    #     class 4       0.47      0.43      0.45      1213
    #     class 5       0.67      0.74      0.70       660
    #     class 6       0.48      0.50      0.49      1139
    #
    # avg / total       0.55      0.55      0.55      6354


    #          precision    recall  f1-score   support
    # class 0       0.70      0.64      0.67       100
    # class 1       0.92      0.78      0.84       100
    # class 2       0.81      0.87      0.84       100
    # class 3       0.97      0.88      0.92       100
    # class 4       0.87      0.98      0.92       100
    # class 5       0.77      0.82      0.80       100
    # class 6       0.84      0.89      0.86       100
    # avg / total       0.84      0.84      0.84       700


    # ours
    #              precision    recall  f1-score   support
    # class 0       0.75      0.65      0.70       100
    # class 1       0.84      0.80      0.82       100
    # class 2       0.87      0.82      0.85       100
    # class 3       0.92      0.96      0.94       100
    # class 4       0.86      0.93      0.89       100
    # class 5       0.76      0.76      0.76       100
    # class 6       0.85      0.95      0.90       100
    # avg / total       0.84      0.84      0.84       700

    # alexnet
    #              precision    recall  f1-score   support
    # class 0       0.59      0.59      0.59       100
    # class 1       0.86      0.76      0.81       100
    # class 2       0.87      0.77      0.81       100
    # class 3       0.88      0.89      0.89       100
    # class 4       0.84      0.85      0.85       100
    # class 5       0.57      0.66      0.61       100
    # class 6       0.80      0.85      0.83       100
    # avg / total       0.77      0.77      0.77       700

    # lenet
    #              precision    recall  f1-score   support
    # class 0       0.55      0.57      0.56       100
    # class 1       0.79      0.80      0.80       100
    # class 2       0.82      0.67      0.74       100
    # class 3       0.92      0.78      0.84       100
    # class 4       0.67      0.85      0.75       100
    # class 5       0.67      0.56      0.61       100
    # class 6       0.72      0.85      0.78       100
    # avg / total       0.73      0.73      0.73       700

    # ours
    #              precision    recall  f1-score   support
    # class 0       0.67      0.66      0.66       100
    # class 1       0.93      0.87      0.90       100
    # class 2       0.84      0.90      0.87       100
    # class 3       0.94      0.93      0.93       100
    # class 4       0.89      0.98      0.93       100
    # class 5       0.81      0.74      0.77       100
    # class 6       0.86      0.86      0.86       100
    # avg / total       0.85      0.85      0.85       700

    #              precision    recall  f1-score   support
    # class 0       0.76      0.65      0.70       100
    # class 1       0.90      0.79      0.84       100
    # class 2       0.86      0.93      0.89       100
    # class 3       0.93      0.93      0.93       100
    # class 4       0.84      0.94      0.89       100
    # class 5       0.78      0.76      0.77       100
    # class 6       0.85      0.92      0.88       100
    # avg / total       0.84      0.85      0.84       700

# ###label :0
# recall:0.65    precision:0.6565656565656566    similarity:0.9014285714285715   F1_measure:0.6532663316582915   accuracy:0.9014285714285715
# =========================================================================
# ###label :1
# recall:0.84    precision:0.8571428571428571    similarity:0.9571428571428572   F1_measure:0.8484848484848485   accuracy:0.9571428571428572
# =========================================================================
# ###label :2
# recall:0.87    precision:0.87    similarity:0.9628571428571429   F1_measure:0.87   accuracy:0.9628571428571429
# =========================================================================
# ###label :3
# recall:0.92    precision:0.9484536082474226    similarity:0.9814285714285714   F1_measure:0.9340101522842639   accuracy:0.9814285714285714
# =========================================================================
# ###label :4
# recall:0.93    precision:0.8532110091743119    similarity:0.9671428571428572   F1_measure:0.8899521531100477   accuracy:0.9671428571428572
# =========================================================================
# ###label :5
# recall:0.76    precision:0.7676767676767676    similarity:0.9328571428571428   F1_measure:0.763819095477387   accuracy:0.9328571428571428
# =========================================================================
# ###label :6
# recall:0.84    precision:0.8571428571428571    similarity:0.9571428571428572   F1_measure:0.8484848484848485   accuracy:0.9571428571428572
# =========================================================================
