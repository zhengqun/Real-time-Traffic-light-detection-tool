import yaml
from svm_train import load_descriptor
from src.detectors import *
from src.helpers import read_directory_images, cutoff_lower,timeit,extract_window
from sklearn.externals import joblib
from ast import literal_eval
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf

# path where the resultant images need to be saved
result_image_path = 'C:/Users/gvadakku/Desktop/final_software_cnn_play/images/test_result_with_cnn/'
# path where the input video is located
input_video_path = 'N:/Giri/darkflow/SYNCAR_TestfeldDresden.mp4'
# Learning rate
LR = 1e-3
# Image resize parameters as per CNN input image size
IMG_SIZE_W, IMG_SIZE_H = 32,64
model_name = 'C:/Users/gvadakku/Desktop/final_software_cnn_play/cnn/traffic_light-0.001-6conv-basic.model'

# Loading the model layout,train the 6-layer CNN prior to using this script final_run.py
tf.reset_default_graph()
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


def load_detector(setting):
    return {
        'colordetector_hsv': ColorDetector_hsv.from_config_file(setting['colordetector_hsv']),
    }.get(setting['run']['detector'], 'colordetector_hsv')


with open("config.yaml", "r") as stream:
    settings = yaml.load(stream)

detector = load_detector(settings)
descriptor = load_descriptor(settings)
classifier = joblib.load(settings['run']['classifier_location'])

win_size = literal_eval(settings['run']['window_size'])
x_offset = win_size[0] / 2
y_offset = win_size[1] / 2
# In case we need a window double the input given in config file say (64*128) as crop size
# window_scaled = [int(x_offset * 4), int(y_offset * 4)]


def cnn_apply(list_lights,image,window_size):
    """
    Function to apply CNN for classifying the TL states on the seeds provided by SVM as potential TL candidates
    :param list_lights: seed coordinates from the output of SVM
    :param image: the original input image
    :param window_size: can be the one used in SVM(32*64) or scaled (64*128)
    :return: list containing the TL coordinates with state
    """
    return_list=[]
    for (x, y) in list_lights:
        window = extract_window(image, (x, y), window_size)
        if window is None:
            return return_list
        # Resizing from (64*128) as crop size input to CNN standard train size(32,64)
        # grey_window = cv2.resize(window, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
        r_grey_window = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY)
        data = r_grey_window.reshape(IMG_SIZE_W, IMG_SIZE_H, 1)
        model_out = model.predict([data])[0]
        if model_out[0] > 0.7:  # Green
            return_list.append([x,y,1])
        elif model_out[1] > 0.7:  # Red
            return_list.append([x, y, 2])
        elif model_out[2] > 0.7:  # Background
            return_list.append([x, y, 3])

    return return_list


@timeit
def image_main():
    """Default function to run the HOG+SVM+CNN TL detector on the images located in the image_directory of run
    configuration"""
    images = read_directory_images(settings['run']['image_directory'], extension='.jpg')
    i = 0
    for image in images:
        top_half = cutoff_lower(image, 0.5)
        lights = classifier.run_detector(detector, top_half, win_size)
        final = cnn_apply(lights,image,win_size)
        # final = cnn_apply(lights,image,window_scaled)
        for (x, y, z) in final:
            if z==1:
                cv2.rectangle(image, (int(x - x_offset), int(y - y_offset)), (int(x + x_offset), int(y + y_offset)), (0, 255, 0), 2)
                print('Green')
            if z == 2:
                cv2.rectangle(image, (int(x - x_offset), int(y - y_offset)), (int(x + x_offset), int(y + y_offset)),
                              (0, 0, 255), 2)
                print('Red')
        i = i + 1
        cv2.imwrite(result_image_path+'img'+str(i)+'.jpg', image)


@timeit
def video():
    """Function to run the HOG+SVM+CNN TL detector on the video located in the path mentioned above(input_video_path)"""
    cap = cv2.VideoCapture(input_video_path)
    out = cv2.VideoWriter('result_video_cnn.avi', -1, 30.0, (1920, 1080))
    # depends on the input video settings('output_video_name',fourcc(-1 gives codec selection),fps(29.0),
    # frameSize(1920,1080))
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is True:
            top_half = cutoff_lower(frame, 0.50)
            lights = classifier.run_detector(detector, top_half, win_size)
            final = cnn_apply(lights, top_half, win_size)
            # final = cnn_apply(lights,image,window_scaled)
            for (x, y, z) in final:
                if z == 1:
                    cv2.rectangle(frame, (int(x - x_offset), int(y - y_offset)), (int(x + x_offset), int(y + y_offset)),
                                  (0, 255, 0), 2)
                    print('Green')
                if z == 2:
                    cv2.rectangle(frame, (int(x - x_offset), int(y - y_offset)), (int(x + x_offset), int(y + y_offset)),
                                  (0, 0, 255), 2)
                    print('Red')
            out.write(frame)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    cnn_meta_path= model_name+str('.meta')
    if os.path.exists(cnn_meta_path):
        model.load(model_name)
        print('model loaded!')
        image_main()
        # video()


#424.65 sec to run with CNN on the video syncar(4500 frames)