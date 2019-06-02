import yaml
from svm_train import load_descriptor
from src.detectors import *
from src.helpers import read_directory_images, cutoff_lower,timeit
from sklearn.externals import joblib
from ast import literal_eval
import cv2

# path where the resultant images need to be saved
result_image_path='C:/Users/gvadakku/Desktop/final_software_cnn_play/images/test_result_svm/'
# path where the input video is located
input_video_path='N:/Giri/darkflow/SYNCAR_TestfeldDresden.mp4'


def load_detector(setting):
    """To load the detector settings from the config file,detector type needs to be changed in run_detector() in svm.py
    It can be hsv_color_segmenation or rgb_spotlight_segmenation"""
    return {
        'colordetector_hsv': ColorDetector_hsv.from_config_file(setting['colordetector_hsv']),
    }.get(setting['run']['detector'], 'colordetector_hsv')


# obtain other configuration settings for run from config.yaml
with open("config.yaml", "r") as stream:
    settings = yaml.load(stream)

detector = load_detector(settings)
descriptor = load_descriptor(settings)
classifier = joblib.load(settings['run']['classifier_location'])

# Grab the shape of the first image in the directory to determine the size of the heatmap

win_size = literal_eval(settings['run']['window_size'])
x_offset = win_size[0] / 2
y_offset = win_size[1] / 2


@timeit
def image_main():
    """Default function to run the HOG+SVM TL detector on the images located in the image_directory of run
    configuration"""
    images = read_directory_images(settings['run']['image_directory'], extension='.jpg')
    i=0
    for image in images:
        top_half = cutoff_lower(image, 0.5)
        lights = classifier.run_detector(detector, top_half, win_size)
        for (x, y) in lights:
            print(x, y)
            cv2.rectangle(image, (int(x - x_offset), int(y - y_offset)), (int(x + x_offset), int(y + y_offset)), (0, 255, 0), 2)

        cv2.imwrite(result_image_path+'img'+str(i)+'.jpg',image)
        i=i+1


@timeit
def video():
    """Function to run the HOG+SVM TL detector on the video located in the path mentioned above(input_video_path)"""
    cap = cv2.VideoCapture(input_video_path)
    out = cv2.VideoWriter('result_video_svm.avi', -1, 29.0, (1920, 1080))
    # depends on the input video settings('output_video_name',fourcc(-1 gives codec selection),fps(29.0),
    # frameSize(1920,1080))
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret is True:
            top_half = cutoff_lower(frame, 0.50)
            lights = classifier.run_detector(detector, top_half, win_size)
            for (x, y) in lights:
                print(x, y)
                cv2.rectangle(frame, (int(x - x_offset), int(y - y_offset)), (int(x + x_offset), int(y + y_offset)),
                              (0, 255, 0), 2)
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
    image_main()
    # video()
