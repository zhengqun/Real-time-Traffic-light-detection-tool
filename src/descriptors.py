import cv2
import numpy as np
from matplotlib import pyplot as plt
from ast import literal_eval
from skimage.feature import hog


class HogDescriptor:
    """
    To compute the HOG features of a given image,all the parameters need to be supplied at config file,it uses L2-Hys
    block normalisation while computing. compute() computes the HOG descriptor while
    display() is just for visualizing the features"""

    def __init__(self, block_size, cell_size, orientations):
        self.block_size = block_size
        self.cell_size = cell_size
        self.orientations = orientations

    def compute(self, image, multichannel=True, visualize=False):
        return hog(image, self.orientations, pixels_per_cell=self.cell_size, cells_per_block=self.block_size,
                   multichannel=multichannel, visualize=visualize, block_norm='L2-Hys')

    def display(self, image):
        fd, hog_image = self.compute(image, visualize=True)
        plt.imshow(hog_image, cmap='gray')
        plt.show()

    @classmethod
    def from_config_file(cls, settings):
        return cls(literal_eval(settings['block_size']), literal_eval(settings['cell_size']),
                   int(settings['orientations']))

    def __repr__(self):
        return " Block Size: {0} \n Cell Size: {1} \n Orientations: {2}" \
               .format(self.block_size, self.cell_size, self.orientations)


