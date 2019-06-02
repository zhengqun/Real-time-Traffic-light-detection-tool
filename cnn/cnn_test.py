import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

LR = 1e-3
TEST_DIR = "C:\\Users\\gvadakku\\tensorflow1\\software\\images\\cnn\\test"
MODEL_NAME = 'traffic_light-{}-{}.model'.format(LR, '6conv-basic')

import tensorflow as tf

IMG_SIZE_W, IMG_SIZE_H=32,64
tf.reset_default_graph()
test_data = np.load('train_data.npy')
test = test_data[-500:]
test_x = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE_W, IMG_SIZE_H, 1)
test_y = [i[1] for i in test]

convnet = input_data(shape=[None, IMG_SIZE_W, IMG_SIZE_H, 1], name='input')

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 3, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR,
                     loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')
fig = plt.figure()
cv2.waitKey(2000)
#cv2.imwrite("detected-boxes.jpg", fig)

if os.path.exists('C:/Users/gvadakku/tensorflow1/software/{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model loaded!')
    for num, data in enumerate(test[:1]):

        img_num = data[1]
        img_data = data[0]

        y = fig.add_subplot(6, 5, num + 1)
        orig = img_data
        data = img_data.reshape(IMG_SIZE_W, IMG_SIZE_H, 1)

        model_out = model.predict([data])[0]
        if model_out[0] > 0.7:
            str_label = 'Green'
        elif model_out[1] > 0.7:
            str_label = 'Red'
        elif model_out[2] > 0.7:
            str_label = 'background'

        y.imshow(orig)
        plt.title(str_label)
        y.axes.get_xaxis().set_visible(False)
        y.axes.get_yaxis().set_visible(False)
    plt.show()
    plt.savefig('results.jpg')