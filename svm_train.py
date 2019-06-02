import yaml
from src import helpers,descriptors,svm,evaluate
from sklearn.externals import joblib
from ast import literal_eval
import numpy as np

# svm test data path to get the matrices like accuracy,recall and precision
svm_test_path='C:/Users/gvadakku/Desktop/final_software_cnn_play/images/svm_data/svm_evaluate/'


def load_descriptor(settings):
    """To load the descriptor settings from the config file,only HOG is supported """

    return {
        'hog': descriptors.HogDescriptor.from_config_file(settings['hog']),
    }.get(settings['train']['descriptor'], 'hog')    # Default to HOG for invalid input


if __name__ == "__main__":
    """Open the config file to load the training configuration,SVM train and test inputs,train and test on the given 
    data,all the paths need to be mentioned in the config file except the svm test data path which is mentioned above,
    image inputs in .jpg only"""

    with open("config.yaml", "r") as stream:
        settings = yaml.load(stream)

    descriptor = load_descriptor(settings)
    classifier = svm.SVM(descriptor, settings['svm']['C'])

    print("Descriptor Settings \n" + str(descriptor))
    print("Classifier Settings \n" + str(classifier))
    print("Reading in the images...")

    positive_images = helpers.read_directory_images(settings['train']['positive_image_directory'], extension='.jpg')
    negative_images = helpers.read_directory_images(settings['train']['negative_image_directory'], extension='.jpg')

    training_size = literal_eval(settings['train']['window_size'])
    positive_images = helpers.resize_images(list(positive_images), training_size)
    negative_images = helpers.resize_images(list(negative_images), training_size)
    print("Total positive images: {}".format(len(positive_images)))
    print("Total negative images: {}".format(len(negative_images)))
    images = np.concatenate((positive_images, negative_images))

    # Set up the labels for binary classification
    labels = np.array([1] * len(positive_images) + [0] * len(negative_images))
    print(labels)
    print("Starting training...")
    classifier.train(images, labels)
    joblib.dump(classifier, settings['train']['outfile'])

    test_folder = helpers.read_directory_images(svm_test_path,extension='.jpg')
    test_images = helpers.resize_images(list(test_folder), training_size)
    classifier = joblib.load(settings['run']['classifier_location'])
    labels = np.array([1] * 130)
    print(str(classifier))
    evaluate.evaluate_model(classifier, test_images, labels)
    print()

