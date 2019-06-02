import cv2
import numpy as np


class ColorDetector_hsv:
    """
    Detector class,this is the first filter stage on the input image.hsv_color_segmenation() is the prime filter.other
    filters are just included for reference.all filters have similar config parameters
    """

    def __init__(self, max_size, kernel_size):
        self.max_size = max_size    # max_size for watershed regions
        self.kernel = kernel_size   # dilation filter kernel size

    def hsv_color_segmentation(self,image):
        """Default method for 1st stage filtering due to better speed.
        Converts the image into hsv,apply the R,G,Y masks for TL objects,thresholds and segments to obtain key
        coloured regions which may be possible TL candidates using watershed segmentation
        :param image:pass the upper n% fo the input image
        :return:list of marker coordinates(x,y) corresponding to the centre of candidate regions
        """
        image = cv2.GaussianBlur(image, (3, 3), 0)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 150, 90])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 150, 90])
        upper_red2 = np.array([180, 255, 255])
        lower_green = np.array([40, 120, 120])
        upper_green = np.array([90, 255, 255])
        lower_yellow = np.array([15, 150, 150])
        upper_yellow = np.array([35, 255, 255])
        # hsv threshold values for red,green and yellow
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        maskg = cv2.inRange(hsv, lower_green, upper_green)
        masky = cv2.inRange(hsv, lower_yellow, upper_yellow)
        maskr = cv2.add(mask1, mask2)
        final_mask = maskr + maskg+masky
        segmented_res = cv2.bitwise_and(hsv, hsv, mask=final_mask)
        hue, saturation, value = cv2.split(segmented_res)
        dilated_image = cv2.dilate(saturation, np.ones((self.kernel, self.kernel)))
        retval, thresholded = cv2.threshold(dilated_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # otsu does not need a threshold value to be supplied,it calculates it for bimodal images
        median_filtered = cv2.medianBlur(thresholded, 3)
        dist_transform = cv2.distanceTransform(np.uint8(median_filtered), cv2.DIST_L2, 5)
        ret, markers = cv2.connectedComponents(np.uint8(dist_transform))
        markers += 1
        watershed_image = cv2.watershed(hsv, markers)
        # Grab the marker values and how many times they occur
        values, counts = np.unique(watershed_image, return_counts=True)
        # Get the indices of where the segments are under the max size
        segment_indices = np.where(counts <= self.max_size)
        markers = values[segment_indices]
        for marker in markers:
            y_coordinates, x_coordinates = np.where(watershed_image == marker)
            yield int(np.median(x_coordinates)), int(np.median(y_coordinates))

    def rgb_spotlight_segmenation(self, image):
        """Optinoal method for first stage filtering
        Applies tophat morphology on the top n%of the image,thresholds and segments to obtain key
        coloured regions which may be possible TL candidates using watershed segmentation
        :param image: pass the upper n% fo the input image
        :return: list of marker coordinates(x,y) corresponding to the centre of candidate regions
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        kernel = np.ones((9, 9), dtype=int)
        threshold = image.max() / 4
        tophat_image = cv2.morphologyEx(gray_image, cv2.MORPH_TOPHAT, kernel)
        ret, thresh = cv2.threshold(tophat_image, threshold, 255, cv2.THRESH_BINARY)
        # Watershed region growing algorithm with the spotlights as the seeds
        dist_transform = cv2.distanceTransform(np.uint8(thresh), cv2.DIST_L2, 5)
        ret, markers = cv2.connectedComponents(np.uint8(dist_transform))
        # Make sure the background is not 0
        markers += 1
        watershed_image = cv2.watershed(image, markers)
        # Grab the marker values and how many times they occur
        values, counts = np.unique(watershed_image, return_counts=True)
        # Get the indices of where the segments are under the max size
        segment_indices = np.where(counts <= self.max_size)
        markers = values[segment_indices]
        # Get the median coordinates of the markers (roughly the center)
        coordinates = []
        for marker in markers:
            y_coordinates, x_coordinates = np.where(watershed_image == marker)
            yield int(np.median(x_coordinates)), int(np.median(y_coordinates))

    '''Naive method-blob detection,not found to be effective fro TL objects 
    def blob_dect(self,image):
        im = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        # Set up the detector with default parameters.
        detector = cv2.SimpleBlobDetector()
        keypoints = detector.detect(im)SSSS
        if keypoints is not None:
            return True
        else:
            return False'''

    def display_roi(self, image, window_size):
        """ Construct windows around the center of the identified ROI and display the image """
        mask = np.zeros(image.shape, dtype=np.uint8)
        x_offset = int((window_size[0] - 1) / 2)
        y_offset = int((window_size[1] - 1) / 2)

        for (x, y) in self.hsv_color_segmenation(image):
            x_min, x_max = x - x_offset, x + x_offset
            y_min, y_max = y - y_offset, y + y_offset
            mask[y_min:y_max, x_min:x_max] = 1

        display_img = np.zeros(image.shape, dtype=np.uint8)
        display_img[mask == 1] = image[mask == 1]

    @classmethod
    def from_config_file(cls, settings):
        return cls(int(settings['max_size']), int(settings['kernel_size']))





