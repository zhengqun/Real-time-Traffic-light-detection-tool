from sklearn.svm import SVC
from sklearn import preprocessing
from src import helpers
import cv2


class SVM:
    """
    SVM classifier: train()-takes the train images with labels and returns the model_fit for dumbing
                    predict()-takes the test image and outputs the predicted label(TL or Background)
                    run_detector()-extracts the  detected candidate region,passes the crop for predict() provided the
                                   crop is valid and yields the the (x,y) coordinates for crops with possible TL objects
    """
    def __init__(self, descriptor, c=1.0):
        """
        C-Support Vector Classification
        :param descriptor: Feature Descriptor Object that converts images into vectors
        :param C: Penalty parameter C of the error term
        """
        self.c = c
        self.clf = SVC(c)
        self.descriptor = descriptor
        self.scaler = preprocessing.StandardScaler()

    def train(self, images, labels):
        features = [self.descriptor.compute(img) for img in images]
        features = self.scaler.fit(features).transform(features)
        self.clf.fit(features, labels)
        return self.clf

    def predict(self, image):
        fd = self.descriptor.compute(image).reshape(1, -1)
        fd = self.scaler.transform(fd)
        return self.clf.predict(fd)[0]

    def predict_all(self, images):
        return [self.predict(image) for image in images]

    def run_sliding_window(self, image, win_size, step_size):
        for (x, y, window) in helpers.sliding_window(image, win_size, step_size):
            if self.predict(window):
                yield (x, y)

    def run_detector(self, detector, image, win_size):
        for (x, y) in detector.hsv_color_segmentation(image):
            window = helpers.extract_window(image, (x, y), win_size)
            #if window is not None and detector.blob_dect(window):
            if window is not None and self.predict(window):
                yield (x, y)

    def __repr__(self):
        return "SVM with C Penalty: {}".format(self.c)
