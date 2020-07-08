from __future__ import division, absolute_import
import re
import numpy as np
from dataset_loader import DatasetLoader
from constants import *
from os.path import isfile, join
import random
import sys
from keras.models import Model, load_model
from keras.layers import Input, Dense, Flatten, merge
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dropout
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D
from keras.optimizers import Adam
from numpy import tile
import cv2


class EmotionRecognition:
    def __init__(self):
        pass
        # self.dataset = DatasetLoader()

    def build_network(self):
        # print('[+] Building CNN')
        # self.model = Sequential()
        # # input: 100x100 images with 3 channels -> (3, 100, 100) tensors.
        # # this applies 32 convolution filters of size 3x3 each.
        # self.model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(48, 48, 1)))
        # self.model.add(Activation('relu'))
        # self.model.add(Convolution2D(32, 3, 3))
        # self.model.add(Activation('relu'))
        # self.model.add(MaxPooling2D(pool_size=(2, 2)))
        # self.model.add(Dropout(0.25))
        #
        # self.model.add(Convolution2D(64, 3, 3, border_mode='same'))
        # self.model.add(Activation('relu'))
        # self.model.add(Convolution2D(64, 3, 3))
        # self.model.add(Activation('relu'))
        # self.model.add(MaxPooling2D(pool_size=(2, 2)))
        # self.model.add(Dropout(0.25))
        #
        # self.model.add(Flatten())
        # # Note: Keras does automatic shape inference.
        # self.model.add(Dense(256))
        # self.model.add(Activation('relu'))
        # self.model.add(Dropout(0.5))
        #
        # self.model.add(Dense(7))
        # self.model.add(Activation('softmax'))
        # sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
        # self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['acc'])

        self.mnist_input = Input(shape=(128, 128, 1), name='input')
        self.conv1 = Conv2D(64, kernel_size=3, activation='relu', name='conv1')(self.mnist_input)
        self.pool1 = MaxPool2D(pool_size=(5, 5), strides=2, name='pool1')(self.conv1)
        self.conv2 = Conv2D(128, kernel_size=2, activation='relu', name='conv2')(self.pool1)
        self.pool2 = MaxPool2D(pool_size=(10, 10), strides=1, name='pool2')(self.conv2)
        self.conv3 = Conv2D(254, kernel_size=6, activation='relu', name='conv3')(self.pool2)
        # self.conv1_1 = Conv2D(128, kernel_size=5, strides=3, activation='relu', name='conv1_1')(self.mnist_input)
        # self.pool1_1 = MaxPool2D(pool_size=(5, 5), strides=2, name='pool1_1')(self.conv1_1)
        # self.concat1 = merge([self.conv3, self.pool1_1], mode='concat', concat_axis=-1)
        # self.conv4 = Conv2D(192, kernel_size=3, activation='relu', name='conv4')(self.concat1)
        self.conv4 = Conv2D(192, kernel_size=2, activation='relu', name='conv4')(self.conv3)
        self.conv5 = Conv2D(256, kernel_size=1, activation='relu', name='conv5')(self.conv4)
        # self.Dropout = Dropout(0.25)
        self.flat1 = Flatten()(self.conv5)
        self.dense1 = Dense(128, activation='relu', name='dense1')(self.flat1)
        self.Dropout = Dropout(0.5)
        self.output = Dense(7, activation='softmax', name='output')(self.dense1)
        self.model = Model(inputs=self.mnist_input, outputs=self.output)
        # sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
        adam = Adam(lr=0.00008, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def build_network_old(self):
        print('[+] Building CNN')
        self.mnist_input = Input(shape=(128, 128, 1), name='input')
        self.conv1 = Conv2D(32, kernel_size=3, activation='relu', name='conv1')(self.mnist_input)
        self.pool1 = MaxPool2D(pool_size=(5, 5), strides=2, name='pool1')(self.conv1)
        self.conv2 = Conv2D(64, kernel_size=6, activation='relu', name='conv2')(self.pool1)
        self.pool2 = MaxPool2D(pool_size=(7, 7), strides=2, name='pool2')(self.conv2)
        self.conv3 = Conv2D(128, kernel_size=6, activation='relu', name='conv3')(self.pool2)
        # self.conv1_1 = Conv2D(64, kernel_size=4, strides=2, activation='relu', name='conv1_1')(self.mnist_input)
        # self.pool1_1 = MaxPool2D(pool_size=(4, 4), strides=2, name='pool1_1')(self.conv1_1)
        self.conv1_1 = Conv2D(64, kernel_size=4, strides=2, activation='relu', name='conv1_1')(self.mnist_input)
        self.pool1_1 = MaxPool2D(pool_size=(4, 4), strides=3, name='pool1_1')(self.conv1_1)
        self.concat1 = merge([self.conv3, self.pool1_1], mode='concat', concat_axis=-1)
        self.conv4 = Conv2D(192, kernel_size=2, activation='relu', name='conv4')(self.concat1)
        self.conv5 = Conv2D(256, kernel_size=1, activation='relu', name='conv5')(self.conv4)
        self.flat1 = Flatten()(self.conv5)
        self.dense1 = Dense(128, activation='relu', name='dense1')(self.flat1)
        self.drop = Dropout(0.15)(self.dense1)
        self.output = Dense(7, activation='softmax', name='output')(self.drop)
        self.model = Model(inputs=self.mnist_input, outputs=self.output)
        # sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
        adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        self.model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer=adam)

    def build_network_alexnet(self):
        self.model = Sequential()
        self.model.add(
                Conv2D(96, (11, 11), strides=(4, 4), input_shape=(128, 128, 1), padding='valid', activation='relu',
                       kernel_initializer='uniform'))
        self.model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        self.model.add(
                Conv2D(256, (5, 5), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        self.model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        self.model.add(
                Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        self.model.add(
                Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        self.model.add(
                Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        self.model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(4096, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(4096, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(7, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

    def build_network_lenet(self):
        model = Sequential()
        model.add(Conv2D(32, (5, 5), strides=(1, 1), input_shape=(128, 128, 1), padding='valid', activation='relu',
                         kernel_initializer='uniform'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, (5, 5), strides=(1, 1), padding='valid', activation='relu', kernel_initializer='uniform'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
        model.add(Dense(7, activation='softmax'))
        model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model = model

    def load_saved_dataset(self):
        # self.dataset.load_from_save()
        print('[+] Dataset found and loaded')

    def start_training(self, images, labels):
        # self.load_saved_dataset()
        self.build_network_old()
        # self.build_network_alexnet()
        # self.build_network_lenet()
        # if self.dataset is None:
        #     self.load_saved_dataset()
        print('[+] Training network')
        print(self.model.summary())
        history = self.model.fit(images, labels, validation_split=0.05, epochs=400, batch_size=32, verbose=2)
        return history

    def predict(self, image):
        if image is None:
            return None
        # re_img = image[2].resize(SIZE_FACE, SIZE_FACE)
        result = image.reshape([-1, SIZE_FACE, SIZE_FACE, 1])
        return self.model.predict(result)

    def save_model(self):
        self.model.save(join(SAVE_DIRECTORY, SAVE_MODEL_FILENAME))
        print('[+] Model trained and saved at ' + SAVE_MODEL_FILENAME)

    def load_model2(self):
        if isfile(join(SAVE_DIRECTORY, SAVE_MODEL_FILENAME)):
            # self.model.load_weights(join(SAVE_DIRECTORY, SAVE_MODEL_FILENAME))
            self.model = load_model(join(SAVE_DIRECTORY, SAVE_MODEL_FILENAME))
            print('[+] Model loaded from ' + SAVE_MODEL_FILENAME)
